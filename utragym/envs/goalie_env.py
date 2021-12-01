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
import enum


class Joints(enum.IntEnum):
    HEAD_1 = 0
    HEAD_2 = 1
    LEFT_ARM_1 = 2
    LEFT_ARM_2 = 3
    LEFT_LEG_1 = 4
    LEFT_LEG_2 = 5
    LEFT_LEG_3 = 6
    LEFT_LEG_4 = 7
    LEFT_LEG_5 = 8
    LEFT_LEG_6 = 9
    RIGHT_ARM_1 = 10
    RIGHT_ARM_2 = 11
    RIGHT_LEG_1 = 12
    RIGHT_LEG_2 = 13
    RIGHT_LEG_3 = 14
    RIGHT_LEG_4 = 15
    RIGHT_LEG_5 = 16
    RIGHT_LEG_6 = 17


class GoalieEnv(BezEnv):

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

        # bez base init state
        pos = self.cfg["env"]["bezInitState"]["pos"]
        rot = self.cfg["env"]["bezInitState"]["rot"]
        v_lin = self.cfg["env"]["bezInitState"]["vLinear"]
        v_ang = self.cfg["env"]["bezInitState"]["vAngular"]

        self.bez_init_state = pos + rot + v_lin + v_ang

        # ball base init state
        pos = self.cfg["env"]["ballInitState"]["pos"]
        rot = self.cfg["env"]["ballInitState"]["rot"]
        v_lin = self.cfg["env"]["ballInitState"]["vLinear"]
        v_ang = self.cfg["env"]["ballInitState"]["vAngular"]
        self.ball_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["readyJointAngles"]  # defaultJointAngles  readyJointAngles

        # other
        self.dt = sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        # for key in self.rew_scales.keys():
        #     self.rew_scales[key] *= self.dt

        # Observation dimension
        self.orn_dim = 2
        self.imu_dim = 6
        self.feet_dim = 8
        self.dof_dim = 18  # or 16 if we remove head
        self.rnn_dim = 1
        self.ball_dim = 2  # ball distance

        # Limits
        self.imu_max_ang_vel = 8.7266
        self.imu_max_lin_acc = 2. * 9.81
        self.AX_12_velocity = (59 / 60) * 2 * np.pi  # Ask sharyar for later - 11.9 rad/s
        self.MX_28_velocity = 2 * np.pi  # 24.5 rad/s
        self.orn_limit = torch.tensor([1.] * self.orn_dim, device='cuda:0')
        self.feet_limit = torch.tensor([1.6] * self.feet_dim, device='cuda:0')
        self.ball_start_limit = torch.tensor([0.3] * self.ball_dim, device='cuda:0')

        # IMU NOISE
        self._IMU_LIN_STDDEV_BIAS = 0.  # 0.02 * _MAX_LIN_ACC
        self._IMU_ANG_STDDEV_BIAS = 0.  # 0.02 * _MAX_ANG_VEL
        self._IMU_LIN_STDDEV = 0.00203 * self.imu_max_lin_acc
        self._IMU_ANG_STDDEV = 0.00804 * self.imu_max_ang_vel

        # FEET NOISE
        self._FEET_FALSE_CHANCE = 0.01

        # Joint angle noise
        self._JOIN_ANGLE_STDDEV = np.pi / 2048
        self._JOIN_VELOCITY_STDDEV = self._JOIN_ANGLE_STDDEV / 120

        self.cfg["env"]["numObservations"] = 54  # self.dof_dim + self.imu_dim + self.orn_dim + self.ball_dim  # 30
        self.cfg["env"]["numActions"] = self.dof_dim

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
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # Update state tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body = gymtorch.wrap_tensor(rigid_body_tensor)
        self.goal = torch.tensor([[1, 0]] * self.num_envs, device='cuda:0')
        self.ball_init = torch.tensor([self.ball_init_state[0:2]] * self.num_envs, device='cuda:0')
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch([self.bez_init_state, self.ball_init_state] * self.num_envs,
                                               device=self.device, requires_grad=False)
        # self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.dof_pos_bez = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel_bez = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.root_pos_bez = self.root_states.view(self.num_envs, 2, 13)[..., 0, 0:3]
        self.root_orient_bez = self.root_states.view(self.num_envs, 2, 13)[..., 0, 3:7]
        self.root_pos_ball = self.root_states.view(self.num_envs, 2, 13)[..., 1, 0:3]
        self.root_orient_ball = self.root_states.view(self.num_envs, 2, 13)[..., 1, 3:7]
        self.root_vel_ball = self.root_states.view(self.num_envs, 2, 13)[..., 1, 7:10]
        self.root_vel_bez = self.rigid_body.view(self.num_envs, -1, 13)[..., 1,
                            7:10]
        self.root_ang_bez = self.rigid_body.view(self.num_envs, -1, 13)[..., 1,
                            10:13]
        self.prev_lin_vel = torch.tensor([0, 0, 0], device='cuda:0')
        self.left_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)[..., 13:17, 0:3]
        self.right_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)[..., 25:29, 0:3]
        # Setting default positions
        self.default_dof_pos = torch.zeros_like(self.dof_pos_bez, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.time_out_buf = torch.zeros_like(self.reset_buf)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
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
        asset_file_bez = "bez/model/soccerbot_box.urdf"
        asset_file_ball = "bez/objects/ball.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file_bez = self.cfg["env"]["asset"].get("assetFileNameBez", asset_file_bez)
            asset_file_ball = self.cfg["env"]["asset"].get("assetFileNameBall", asset_file_ball)

        asset_path = os.path.join(asset_root, asset_file_bez)
        asset_path_ball = os.path.join(asset_root, asset_file_ball)
        asset_root = os.path.dirname(asset_path)
        asset_file_bez = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"]
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]
        asset_options.replace_cylinder_with_capsule = self.cfg["env"]["urdfAsset"]["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = self.cfg["env"]["urdfAsset"]["flip_visual_attachments"]
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = self.cfg["env"]["urdfAsset"]["density"]
        asset_options.angular_damping = self.cfg["env"]["urdfAsset"]["angular_damping"]
        asset_options.linear_damping = self.cfg["env"]["urdfAsset"]["linear_damping"]
        asset_options.armature = self.cfg["env"]["urdfAsset"]["armature"]
        asset_options.thickness = self.cfg["env"]["urdfAsset"]["thickness"]
        asset_options.disable_gravity = self.cfg["env"]["urdfAsset"]["disable_gravity"]

        bez_asset = self.gym.load_asset(self.sim, asset_root, asset_file_bez, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(bez_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(bez_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.bez_init_state[:3])

        self.dof_names = self.gym.get_asset_dof_names(bez_asset)

        actuator_props = self.gym.get_asset_dof_properties(bez_asset)

        motor_efforts = [prop['effort'] for prop in actuator_props]

        self.max_motor_effort = to_torch(motor_efforts, device=self.device)
        self.min_motor_effort = -1 * to_torch(motor_efforts, device=self.device)

        self.num_bodies = self.gym.get_asset_rigid_body_count(bez_asset)
        self.num_dof = self.gym.get_asset_dof_count(bez_asset)
        self.num_joints = self.gym.get_asset_joint_count(bez_asset)

        for i in range(self.num_dof):
            actuator_props['driveMode'][i] = self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"]
            actuator_props['stiffness'][i] = self.Kp
            actuator_props['damping'][i] = self.Kd
            actuator_props["armature"][i] = self.cfg["env"]["urdfAsset"]["armature"]
            actuator_props["velocity"][i] = self.MX_28_velocity

        asset_root_ball = os.path.dirname(asset_path_ball)
        asset_file_ball = os.path.basename(asset_path_ball)

        asset_options_ball = gymapi.AssetOptions()
        asset_options_ball.default_dof_drive_mode = 0
        asset_options_ball.armature = self.cfg["env"]["urdfAsset"]["armature"]

        ball_asset = self.gym.load_asset(self.sim, asset_root_ball, asset_file_ball, asset_options_ball)

        start_pose_ball = gymapi.Transform()
        start_pose_ball.p = gymapi.Vec3(*self.ball_init_state[:3])

        # compute aggregate size
        num_bez_bodies = self.gym.get_asset_rigid_body_count(bez_asset)
        num_bez_shapes = self.gym.get_asset_rigid_shape_count(bez_asset)
        num_ball_bodies = self.gym.get_asset_rigid_body_count(ball_asset)
        num_ball_shapes = self.gym.get_asset_rigid_shape_count(ball_asset)
        max_agg_bodies = num_bez_bodies + num_ball_bodies
        max_agg_shapes = num_bez_shapes + num_ball_shapes

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.bez_indices = []
        self.ball_indices = []
        self.bez_handles = []
        self.ball_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            bez_handle = self.gym.create_actor(env_ptr, bez_asset, start_pose, "bez", i, 0,
                                               0)  # 1 for no self collision
            self.gym.set_actor_dof_properties(env_ptr, bez_handle, actuator_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, bez_handle)

            bez_idx = self.gym.get_actor_index(env_ptr, bez_handle, gymapi.DOMAIN_SIM)
            self.bez_indices.append(bez_idx)

            # debugging
            # name = self.gym.get_actor_name(env_ptr, bez_handle)
            #
            # body_names = self.gym.get_actor_rigid_body_names(env_ptr, bez_handle)
            # body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, bez_handle)
            #
            # joint_names = self.gym.get_actor_joint_names(env_ptr, bez_handle)
            # joint_dict = self.gym.get_actor_joint_dict(env_ptr, bez_handle)
            #
            # dof_names = self.gym.get_actor_dof_names(env_ptr, bez_handle)
            # dof_dict = self.gym.get_actor_dof_dict(env_ptr, bez_handle)
            #
            # print()
            # print("===== Actor: %s =======================================" % name)
            #
            # print("\nBodies")
            # print(body_names)
            # print(body_dict)
            #
            # print("\nJoints")
            # print(joint_names)
            # print(joint_dict)
            #
            # print("\n Degrees Of Freedom (DOFs)")
            # print(dof_names)
            # print(dof_dict)

            ball_handle = self.gym.create_actor(env_ptr, ball_asset, start_pose_ball, "ball", i, 0,
                                                0)
            self.gym.enable_actor_dof_force_sensors(env_ptr, ball_handle)
            ball_idx = self.gym.get_actor_index(env_ptr, ball_handle, gymapi.DOMAIN_SIM)
            self.ball_indices.append(ball_idx)

            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.bez_handles.append(bez_handle)
            self.ball_handles.append(ball_handle)

        self.bez_indices = to_torch(self.bez_indices, dtype=torch.long, device=self.device)
        self.ball_indices = to_torch(self.ball_indices, dtype=torch.long, device=self.device)
        self.dof_pos_limits_lower = []
        self.dof_pos_limits_upper = []

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, bez_handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_pos_limits_lower.append(dof_prop['upper'][j])
                self.dof_pos_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_pos_limits_lower.append(dof_prop['lower'][j])
                self.dof_pos_limits_upper.append(dof_prop['upper'][j])

        self.dof_pos_limits_lower = to_torch(self.dof_pos_limits_lower, device=self.device)
        self.dof_pos_limits_upper = to_torch(self.dof_pos_limits_upper, device=self.device)

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        self.actions = actions.clone().to(self.device)

        # computed_torque = self.Kp*(self.actions-self.dof_pos)
        # computed_torque -= self.Kd*(self.actions-self.dof_vel)
        # applied_torque = saturate(
        #     computed_torque,
        #     lower=self.min_motor_effort,
        #     upper=self.max_motor_effort
        # )
        # action = torch.zeros(applied_torque.size(), dtype=torch.float, device=self.device)
        # action[0][2] = -10#self.Kp*(3-self.dof_pos[0][2])-self.Kd*(self.dof_vel[0][2])#applied_torque[0][2]
        # self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(action))
        # self.actions[0][2] = 10
        targets = self.actions + self.default_dof_pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations

        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def imu(self):
        lin_vel = self.root_vel_bez
        lin_acc = (lin_vel - self.prev_lin_vel) / (self.dt)
        lin_acc -= self.gravity_vec

        rot_mat = quaternion_to_matrix(self.root_orient_bez)

        lin_acc_transform = torch.matmul(rot_mat, lin_acc.reshape((self.num_envs, -1, 1))).reshape(
            (self.num_envs, -1,))  # nx3x3 * n*3*1
        ang_vel = self.root_ang_bez

        self.prev_lin_vel = lin_vel

        lin_acc_clipped = torch.clamp(lin_acc_transform, -self.imu_max_lin_acc, self.imu_max_lin_acc)
        ang_vel_clipped = torch.clamp(ang_vel, -self.imu_max_ang_vel, self.imu_max_ang_vel)

        imu_val = torch.cat([lin_acc_clipped, ang_vel_clipped], 1)
        return imu_val

    def off_orn(self):
        distance_to_goal = torch.sub(self.goal, self.root_pos_bez[..., 0:2])
        distance_to_goal_norm = torch.reshape(torch.linalg.norm(distance_to_goal, dim=1), (-1, 1))
        distance_unit_vec = torch.div(distance_to_goal, distance_to_goal_norm)

        vec = torch.zeros(self.num_envs, 2, device=self.device)
        x = self.root_orient_bez[..., 0]
        y = self.root_orient_bez[..., 1]
        z = self.root_orient_bez[..., 2]
        w = self.root_orient_bez[..., 3]
        wz = torch.mul(w, z)
        xy = torch.mul(x, y)
        yy = torch.mul(y, y)
        zz = torch.mul(z, z)
        wz_xy = torch.add(wz, xy)
        wz_xy_2 = torch.mul(2, wz_xy)
        yy_zz = torch.add(yy, zz)
        yy_zz_2 = torch.mul(2, yy_zz)
        yy_zz_2_1 = torch.sub(1, yy_zz_2)
        yaw = torch.atan2(wz_xy_2, yy_zz_2_1)
        cos = torch.cos(yaw)
        sin = torch.sin(yaw)
        d2_vect = torch.cat((cos.reshape((-1, 1)), sin.reshape((-1, 1))), dim=-1)
        cos = torch.sum(d2_vect * distance_unit_vec, dim=-1)
        pad = torch.tensor([0], device=self.device)
        distance_unit_vec_3d = torch.nn.functional.pad(input=distance_unit_vec, pad=(0, 1, 0, 0), mode='constant',
                                                       value=0)
        d2_vect_3d = torch.nn.functional.pad(input=d2_vect, pad=(0, 1, 0, 0), mode='constant',
                                             value=0)
        sin = torch.linalg.norm(torch.cross(distance_unit_vec_3d, d2_vect_3d, dim=1), dim=1)
        vec = torch.cat((sin.reshape((-1, 1)), -cos.reshape((-1, 1))), dim=-1)
        # print('vec: ', vec.shape, vec)

        return vec

    def feet(self):
        """
        Checks if 4 corners of the each feet are in contact with ground
        Indicies for looking from above on the feet plates:
          Left         Right
        4-------5    0-------1
        |   ^   |    |   ^   |      ^
        |   |   |    |   |   |      | : forward direction
        |       |    |       |
        6-------7    2-------3
        :return: int array of 8 contact points on both feet, 1: that point is touching the ground -1: otherwise
        """
        forces = torch.tensor(self.num_envs * [[-1.] * self.feet_dim], device=self.device)  # nx8
        ones = torch.ones_like(forces)
        left_pts = torch.linalg.norm(self.left_contact_forces.T, dim=0).T  # nx4
        right_pts = torch.linalg.norm(self.right_contact_forces.T, dim=0).T  # nx4
        pts = torch.cat((left_pts, right_pts), 1)  # nx8
        location = torch.where(pts > 1.0, ones, forces)

        return location

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_bez_reward(
            # tensors
            self.dof_pos_bez,
            self.default_dof_pos,
            self.root_vel_bez,
            self.root_ang_bez,
            self.root_pos_bez,
            self.root_orient_bez,
            self.root_pos_ball,
            self.root_vel_ball,
            self.goal,
            self.ball_init,
            self.reset_buf,
            self.progress_buf,
            # int
            self.max_episode_length,

        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # print(self.dof_pos_bez.shape)
        # print(self.dof_vel_bez.shape)
        # print(self.imu().shape)
        # print(self.off_orn().shape)
        # print('feet: ', self.feet().shape)
        # print('ball_init: ',self.ball_init.shape, self.ball_init)
        self.imu()
        self.obs_buf[:] = compute_bez_observations(
            # tensors
            self.dof_pos_bez,  # 18
            self.dof_vel_bez,  # 18
            self.imu(),  # 6
            self.off_orn(),  # 2
            self.feet(),  # 8
            self.ball_init  # 2
        )

    def reset(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0, 1, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-1, 0, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos_bez[env_ids] = self.default_dof_pos[env_ids]  # * positions_offset
        self.dof_vel_bez[env_ids] = velocities

        bez_indices = torch.unique(torch.cat([self.bez_indices[env_ids],
                                              self.ball_indices[env_ids]]).to(torch.int32))
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(bez_indices), len(bez_indices))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(self.bez_indices.to(dtype=torch.int32)),
                                              len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


#####################################################################
###=========================jit functions=========================###
#####################################################################
# todo
@torch.jit.script
def compute_bez_reward(
        # tensors
        dof_pos_bez: Tensor,
        default_dof_pos: Tensor,
        imu_lin_bez: Tensor,
        imu_ang_bez: Tensor,
        root_pos_bez: Tensor,
        root_orient_bez: Tensor,
        root_pos_ball: Tensor,
        root_vel_ball: Tensor,
        goal: Tensor,
        ball_init: Tensor,
        reset_buf: Tensor,
        progress_buf: Tensor,
        max_episode_length: int,

) -> Tuple[Tensor, Tensor]:  # (reward, reset, feet_in air, feet_air_time, episode sums)

    distance_to_ball = torch.sub(root_pos_ball[..., 0:2], root_pos_bez[..., 0:2])  # nx2
    distance_to_ball_norm = torch.reshape(torch.linalg.norm(distance_to_ball, dim=1), (-1, 1))
    distance_unit_vec = torch.div(distance_to_ball, distance_to_ball_norm)  # 2xn / nx1 = nx2

    velocity_forward_reward = torch.sum(distance_unit_vec * imu_lin_bez[..., 0:2], dim=-1)

    distance_to_goal = torch.sub(goal, root_pos_ball[..., 0:2])
    distance_to_goal_norm = torch.reshape(torch.linalg.norm(distance_to_goal, dim=1), (-1, 1))
    distance_unit_vec = torch.div(distance_to_goal, distance_to_goal_norm)

    ball_velocity_forward_reward = torch.sum(distance_unit_vec * root_vel_ball[..., 0:2], dim=-1)
    ball_velocity_norm = torch.reshape(torch.linalg.norm(root_vel_ball[..., 0:2], dim=1), (-1, 1))

    DESIRED_HEIGHT = 0.27  # seems arbitrary

    # reward
    vel_reward = 0.05 * torch.linalg.norm(imu_ang_bez, dim=1)
    pos_reward = 0.05 * torch.linalg.norm(default_dof_pos - dof_pos_bez, dim=1)
    distance_to_height = DESIRED_HEIGHT - root_pos_bez[..., 2]
    distance_kicked = torch.linalg.norm(torch.sub(root_pos_ball[..., 0:2], ball_init), dim=1)
    # print('here ',vel_reward.shape,pos_reward.shape,distance_to_height.shape,ball_velocity_forward_reward.shape,
    # velocity_forward_reward.shape)

    #  0.2 * ball_velocity_forward_reward - distance_to_height - vel_reward - pos_reward
    vel_pos_reward = torch.sub(vel_reward, pos_reward)
    height_vel_pos_reward = torch.sub(distance_to_height, vel_pos_reward)
    ball_velocity_forward_reward_scaled = torch.mul(ball_velocity_forward_reward, 0.2)
    ball_height_vel_pos_reward = torch.sub(ball_velocity_forward_reward_scaled, height_vel_pos_reward)

    # 0.2 * ball_velocity_forward_reward + 0.05 * velocity_forward_reward - distance_to_height
    velocity_forward_reward_scaled = torch.mul(velocity_forward_reward, 0.05)
    vel_height_reward = torch.sub(velocity_forward_reward_scaled, distance_to_height)
    ball_vel_height_reward = torch.add(ball_velocity_forward_reward_scaled, vel_height_reward)

    reward = torch.where(distance_kicked > 0.3,
                         ball_height_vel_pos_reward,
                         ball_vel_height_reward
                         )
    # print('start: ', reward)
    # Reset
    ones = torch.ones_like(reset_buf)
    reset = torch.zeros_like(reset_buf)

    # =====================================
    # Close to the Goal
    # if torch.linalg.norm(root_pos_ball[..., 0:2] - goal) < 0.05:
    #     print('Close to the Goal')
    # =====================================

    # distance_to_goal_norm = torch.reshape(distance_to_goal_norm, (-1))
    # reset = torch.where(distance_to_goal_norm < 0.05, ones,
    #                     reset)
    # reward = torch.where(distance_to_goal_norm < 0.05,
    #                      torch.ones_like(reward) * 100,
    #                      reward)

    # =====================================
    # Out of Bound
    # if torch.linalg.norm(root_pos_ball[..., 0:2] - goal) > 3 * torch.linalg.norm(goal.to(dtype=torch.float32)):
    #     print('Out of Bound')
    # =====================================

    # goal_norm_scaled = torch.mul(torch.linalg.norm(goal.to(dtype=torch.float32)), 3)
    # reset = torch.where(
    #     distance_to_goal_norm > goal_norm_scaled,
    #     ones,
    #     reset)
    # reward = torch.where(
    #     distance_to_goal_norm > goal_norm_scaled,
    #     torch.ones_like(reward) * -10000,
    #     reward)

    # =====================================
    # Failed to stop the ball
    # if torch.linalg.norm(root_pos_ball[..., 0:2] - goal) > 3 * torch.linalg.norm(goal.to(dtype=torch.float32)):
    #     print('Out of Bound')
    # =====================================

    reset = torch.where(
        distance_to_ball[..., 0] < 0,
        ones,
        reset)
    reward = torch.where(
        distance_to_ball[..., 0] < 0,
        torch.ones_like(reward) * -10000.0,
        reward)

    # =====================================
    # if ball stopped:
    #     print('Ball Stopped')
    # =====================================
    reset = torch.where(ball_velocity_norm[..., 0] < 0.05, ones,
                        reset)

    # =====================================
    # if progress_buf >= max_episode_length:
    #     print('Horizon Ended')
    # =====================================

    # Horizon Ended
    reset = torch.where(progress_buf >= max_episode_length, ones, reset)
    reward = torch.where(progress_buf >= max_episode_length, torch.zeros_like(reward),
                         reward)
    # print(reward)
    return reward, reset


@torch.jit.script
def compute_bez_observations(dof_pos_bez: Tensor,
                             dof_vel_bez: Tensor,  # 18
                             imu: Tensor,  # 6
                             off_orn: Tensor,  # 2
                             feet: Tensor,  # 8
                             ball_init: Tensor  # 2
                             ) -> Tensor:
    obs = torch.cat((dof_pos_bez,
                     dof_vel_bez,  # 18
                     imu,  # 6
                     off_orn,  # 2
                     feet,  # 8
                     ball_init  # 2
                     ), dim=-1)

    return obs
