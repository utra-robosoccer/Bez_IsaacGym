import numpy as np
import os
import time

from utragym.utils.torch_jit_utils import *
from utragym.envs.base.bez_env import BezEnv
from isaacgym import gymtorch
from isaacgym import gymapi

import torch
from torch.tensor import Tensor
from typing import Tuple, Dict


class KickEnv(BezEnv):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.randomize = False
        # Setup
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"] # defaultJointAngles  readyJointAngles

        # other
        self.dt = sim_params.dt
        # self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        # self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        # for key in self.rew_scales.keys():
        #     self.rew_scales[key] *= self.dt

        self.cfg["env"]["numObservations"] = 51
        self.cfg["env"]["numActions"] = 18

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        # camera view
        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body = gymtorch.wrap_tensor(rigid_body_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.root_pos = self.root_states.view(self.num_envs, 13)[..., 0:3]
        self.root_orient = self.root_states.view(self.num_envs, 13)[..., 3:7]
        num_rigid_bodies = int(self.gym.get_sim_rigid_body_count(self.sim) / self.num_envs)
        self.imu_lin = self.rigid_body.view(self.num_envs, num_rigid_bodies, 13)[..., 1,
                       7:10]
        self.imu_ang = self.rigid_body.view(self.num_envs, num_rigid_bodies, 13)[..., 1,
                       10:13]

        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device,
                                                requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.time_out_buf = torch.zeros_like(self.reset_buf)

        self.reset(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = "../../assets"
        asset_file = "urdf/bez/model/soccerbot_box.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"]
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]
        asset_options.replace_cylinder_with_capsule = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]
        asset_options.flip_visual_attachments = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = self.cfg["env"]["urdfAsset"]["density"]
        asset_options.angular_damping = self.cfg["env"]["urdfAsset"]["angular_damping"]
        asset_options.linear_damping = self.cfg["env"]["urdfAsset"]["linear_damping"]
        asset_options.armature = self.cfg["env"]["urdfAsset"]["armature"]
        asset_options.thickness = self.cfg["env"]["urdfAsset"]["thickness"]
        asset_options.disable_gravity = self.cfg["env"]["urdfAsset"]["disable_gravity"]

        bez_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(bez_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(bez_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.dof_names = self.gym.get_asset_dof_names(bez_asset)
        self.base_index = 0

        actuator_props = self.gym.get_asset_dof_properties(bez_asset)

        motor_efforts = [prop['effort'] for prop in actuator_props]

        self.max_motor_effort = to_torch(motor_efforts, device=self.device)
        self.min_motor_effort = -1*to_torch(motor_efforts, device=self.device)

        self.num_bodies = self.gym.get_asset_rigid_body_count(bez_asset)
        self.num_dof = self.gym.get_asset_dof_count(bez_asset)
        self.num_joints = self.gym.get_asset_joint_count(bez_asset)

        for i in range(self.num_dof):
            actuator_props['driveMode'][i] = self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"]
            actuator_props['stiffness'][i] = self.Kp
            actuator_props['damping'][i] = self.Kd

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.bez_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            bez_handle = self.gym.create_actor(env_ptr, bez_asset, start_pose, "bez", i, 1, 1)  # 0 for no self collision
            self.gym.set_actor_dof_properties(env_ptr, bez_handle, actuator_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, bez_handle)
            self.envs.append(env_ptr)
            self.bez_handles.append(bez_handle)

        self.dof_limits_lower = []
        self.dof_limits_upper = []

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, bez_handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.bez_handles[0], "base")

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        self.actions = actions.clone().to(self.device)
        # print('here')
        # print(self.actions)
        # print(self.dof_pos)
        # print(self.actions-self.dof_pos)
        # computed_torque = self.Kp*(self.actions-self.dof_pos)
        # computed_torque -= self.Kd*(self.actions-self.dof_vel)
        # applied_torque = saturate(
        #     computed_torque,
        #     lower=self.min_motor_effort,
        #     upper=self.max_motor_effort
        # )
        # # print(self.min_motor_effort,self.max_motor_effort)
        # action = torch.zeros(applied_torque.size(), dtype=torch.float, device=self.device)
        # action[0][2] = -10#self.Kp*(3-self.dof_pos[0][2])-self.Kd*(self.dof_vel[0][2])#applied_torque[0][2]
        # print(action)
        # print(self.dof_pos)
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(action))
        # self.actions[0][2] = 10
        targets = self.actions #+ self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self, test):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1

        # Turn off for testing
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0 and not test:
            self.reset(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_bez_reward(
            # tensors
            self.root_states,

            self.reset_buf,
            # Dict

            # other
            self.base_index

        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.root_pos = self.root_states.view(self.num_envs, 13)[..., 0:3]
        self.root_orient = self.root_states.view(self.num_envs, 13)[..., 3:7]
        num_rigid_bodies = int(self.gym.get_sim_rigid_body_count(self.sim) / self.num_envs)
        self.imu_lin = self.rigid_body.view(self.num_envs, num_rigid_bodies, 13)[..., 1,
                       7:10]
        self.imu_ang = self.rigid_body.view(self.num_envs, num_rigid_bodies, 13)[..., 1,
                       10:13]

        # self.obs_buf[:] = compute_bez_observations(  # tensors
        #     self.root_states,
        #     self.dof_pos,
        #     self.default_dof_pos,
        #     self.dof_vel,
        #     self.actions
        #
        # )

    def reset(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0, 1, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-1, 0, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] #* positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_bez_reward(
        # tensors
        root_states: Tensor,
        reset_buf: Tensor,
        # Dict

        # other
        base_index: int,

) -> Tuple[Tensor, Tensor]:  # (reward, reset, feet_in air, feet_air_time, episode sums)

    return torch.ones_like(reset_buf), torch.ones_like(reset_buf)


@torch.jit.script
def compute_bez_observations(root_states: Tensor,
                             commands: Tensor,
                             dof_pos: Tensor,
                             default_dof_pos: Tensor,
                             dof_vel: Tensor,
                             gravity_vec: Tensor,
                             actions: Tensor,
                             lin_vel_scale: float,
                             ang_vel_scale: float,
                             dof_pos_scale: float,
                             dof_vel_scale: float
                             ) -> Tensor:
    # base_position = root_states[:, 0:3]
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands * torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False,
                                              device=commands.device)

    obs = torch.cat((base_lin_vel,
                     base_ang_vel,
                     projected_gravity,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel * dof_vel_scale,
                     actions
                     ), dim=-1)

    return obs
