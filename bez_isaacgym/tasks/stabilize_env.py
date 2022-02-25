import os
import time
import math
import sys

from matplotlib import pyplot as plt

from utils.torch_jit_utils import *
from .base.vec_task import VecTask
from isaacgym import gymtorch
from isaacgym import gymapi

import torch
from torch._tensor import Tensor
from typing import Tuple, Dict, Any
import enum

import matplotlib

matplotlib.use("TkAgg")


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


class StabilizeEnv(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        # Setup
        self.cfg = cfg

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

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

        # goal state
        goal = self.cfg["env"]["goalState"]["goal"]

        # model choice
        self.cleats = self.cfg["env"]["asset"]["cleats"]

        # debug
        self.debug_rewards = self.cfg["env"]["debug"]["rewards"]

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["readyJointAngles"]  # defaultJointAngles  readyJointAngles
        self.named_joint_limit_high = self.cfg["env"]["JointLimitHigh"]
        self.named_joint_limit_low = self.cfg["env"]["JointLimitLow"]

        # other
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        # Observation dimension
        self.orn_dim = 2
        self.imu_dim = 6
        self.feet_dim = 8
        self.dof_dim = 18  # or 16 if we remove head
        self.rnn_dim = 1

        # Limits
        self.imu_max_ang_vel = 8.7266
        self.imu_max_lin_acc = 2. * 9.81
        self.AX_12_velocity = (59 / 60) * 2 * np.pi  # Ask sharyar for later - 11.9 rad/s
        self.MX_28_velocity = 2 * np.pi  # 24.5 rad/s

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

        # Number of observation and actions
        self.cfg["env"][
            "numObservations"] = self.dof_dim + self.dof_dim + 4  # self.dof_dim + self.dof_dim + self.imu_dim +
        # self.orn_dim + self.feet_dim + self.ball_dim  # 54
        self.cfg["env"]["numActions"] = self.dof_dim

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)

        # simulation parameters
        self.dt = self.sim_params.dt
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        # TODO Implement
        self.orn_limit = torch.tensor([1.] * self.orn_dim, device=self.device)
        self.feet_limit = torch.tensor([1.6] * self.feet_dim, device=self.device)

        # camera view
        if self.viewer is not None:
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

        self.goal = torch.tensor([goal], device=self.device).repeat((self.num_envs, 1))
        self.bez_init_xy = torch.tensor(self.bez_init_state[0:2], device=self.device)

        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch([self.bez_init_state],
                                               device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        self.initial_root_states[:, 7:13] = 0

        self.dof_pos_bez = self.dof_state.view(self.num_envs, self.num_dof, -1)[..., 0]
        self.dof_vel_bez = self.dof_state.view(self.num_envs, self.num_dof, -1)[..., 1]

        self.root_pos_bez = self.root_states.view(self.num_envs, 13)[..., 0:3]

        # Imu link
        self.root_orient_bez = self.rigid_body.view(self.num_envs, -1, 13)[..., 1, 3:7]
        self.root_vel_bez = self.rigid_body.view(self.num_envs, -1, 13)[..., 1, 7:10]
        self.root_ang_bez = self.rigid_body.view(self.num_envs, -1, 13)[..., 1, 10:13]

        self.prev_lin_vel = torch.tensor([[0, 0, 0]], device=self.device).repeat((self.num_envs, 1))

        self.feet = torch.tensor([[-1.] * self.feet_dim], device=self.device).repeat((self.num_envs, 1))

        if self.cleats:
            self.left_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)[..., 13:17,
                                       0:3]
            self.right_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)[..., 25:29,
                                        0:3]
        else:
            self.left_foot_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)[..., 12,
                                            0:3]
            self.right_foot_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)[...,
                                             20, 0:3]

        # self.left_arm_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)[..., 6,
        # 0:3]
        # self.right_arm_contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1,
        # 3)[..., 18,0:3]

        # Setting default positions
        self.default_dof_pos = torch.zeros_like(self.dof_pos_bez, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        self.dof_pos_limits_upper = torch.zeros(self.dof_dim, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        self.dof_pos_limits_lower = torch.zeros(self.dof_dim, dtype=torch.float, device=self.device,
                                                requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle
            self.dof_pos_limits_upper[i] = self.named_joint_limit_high[name]
            self.dof_pos_limits_lower[i] = self.named_joint_limit_low[name]
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # initialize some data used later on
        self.common_step_counter = 0
        self.push_interval = int(1 / self.dt + 0.5)  # TODO
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.gravity_vec = to_torch([0, 0, -9.81], device=self.device).repeat(
            (self.num_envs, 1))

        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.time = []
        self.kick_vel = []
        self.up_proj = []
        self.goal_angle_diff = []
        self.distance_kicked_norm = []
        self.vel_reward_scaled = []
        self.pos_reward_scaled = []
        self.max_kick_velocity = 0.0

        if self.debug_rewards:
            self.fig, self.ax = plt.subplots(2, 3)

            self.fig.show()

            # We need to draw the canvas before we start animating...
            self.fig.canvas.draw()

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        # If randomizing, apply once immediately on startup before the first sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = "../../assets"
        asset_file_bez = "bez/model/soccerbot_box_sensor.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)

            if self.cfg["env"]["asset"]["stl"]:
                if self.cleats:
                    asset_file_bez = self.cfg["env"]["asset"].get("assetFileNameBezStlSensor", asset_file_bez)
                else:
                    asset_file_bez = self.cfg["env"]["asset"].get("assetFileNameBezStl", asset_file_bez)
            else:
                if self.cleats:
                    asset_file_bez = self.cfg["env"]["asset"].get("assetFileNameBezBoxSensor", asset_file_bez)
                else:
                    asset_file_bez = self.cfg["env"]["asset"].get("assetFileNameBezBox", asset_file_bez)

        asset_path = os.path.join(asset_root, asset_file_bez)
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
        # asset_options.override_com = True
        # asset_options.override_inertia = True
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params.resolution = 300000
        # asset_options.vhacd_params.max_convex_hulls = 10
        # asset_options.vhacd_params.max_num_vertices_per_ch = 64

        bez_asset = self.gym.load_asset(self.sim, asset_root, asset_file_bez, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(bez_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(bez_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.bez_init_state[:3])
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
                                           device=self.device)
        self.dof_names = self.gym.get_asset_dof_names(bez_asset)

        actuator_props = self.gym.get_asset_dof_properties(bez_asset)

        self.num_bodies = self.gym.get_asset_rigid_body_count(bez_asset)
        self.num_dof = self.gym.get_asset_dof_count(bez_asset)
        self.num_joints = self.gym.get_asset_joint_count(bez_asset)

        self.actuated_dof_indices = [self.gym.find_asset_dof_index(bez_asset, name) for name in
                                     self.dof_names]

        for i in range(self.num_dof):
            actuator_props['driveMode'][i] = self.cfg["env"]["urdfAsset"]["defaultDofDriveMode"]
            actuator_props['stiffness'][i] = self.Kp
            actuator_props['damping'][i] = self.Kd
            actuator_props["armature"][i] = self.cfg["env"]["urdfAsset"]["armature"]
            actuator_props["velocity"][i] = self.MX_28_velocity
            actuator_props['friction'][i] = 0.1
            actuator_props['effort'][i] = 2.5

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        self.bez_indices = []
        self.bez_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            bez_handle = self.gym.create_actor(env_ptr, bez_asset, start_pose, "bez", i, 0,
                                               0)  # 1 for no self collision
            self.gym.set_actor_dof_properties(env_ptr, bez_handle, actuator_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, bez_handle)

            bez_idx = self.gym.get_actor_index(env_ptr, bez_handle, gymapi.DOMAIN_SIM)
            self.bez_indices.append(bez_idx)

            self.envs.append(env_ptr)
            self.bez_handles.append(bez_handle)

        self.bez_indices = to_torch(self.bez_indices, dtype=torch.long, device=self.device)

        # self.dof_pos_limits_lower = []
        # self.dof_pos_limits_upper = []
        self.dof_vel_limits_upper = []
        self.dof_vel_limits_lower = []

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, bez_handle)
        for j in range(self.num_dof):
            # if dof_prop['lower'][j] > dof_prop['upper'][j]:
            #     self.dof_pos_limits_lower.append(dof_prop['upper'][j])
            #     self.dof_pos_limits_upper.append(dof_prop['lower'][j])
            # else:
            #     self.dof_pos_limits_lower.append(dof_prop['lower'][j])
            #     self.dof_pos_limits_upper.append(dof_prop['upper'][j])

            self.dof_vel_limits_upper.append([self.MX_28_velocity])
            self.dof_vel_limits_lower.append([-self.MX_28_velocity])

        # self.dof_pos_limits_lower = to_torch(self.dof_pos_limits_lower, device=self.device)
        # self.dof_pos_limits_upper = to_torch(self.dof_pos_limits_upper, device=self.device)
        self.dof_vel_limits_upper = to_torch(self.dof_vel_limits_upper, device=self.device)
        self.dof_vel_limits_lower = to_torch(self.dof_vel_limits_lower, device=self.device)

    def pre_physics_step(self, actions):
        # implement pre-physics simulation code here
        #    - e.g. apply actions
        self.actions = actions.clone().to(self.device)
        self.actions[..., 0:2] = 0.0  # Remove head action

        # Position Control
        targets = tensor_clamp(self.actions + self.default_dof_pos, self.dof_pos_limits_lower,
                               self.dof_pos_limits_upper)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

        # Velocity Control
        # targets = tensor_clamp(self.actions, self.dof_vel_limits_upper,
        #                        self.dof_vel_limits_lower)
        # self.gym.set_dof_velocity_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        # implement post-physics simulation code here
        #    - e.g. compute reward, compute observations
        self.progress_buf += 1
        self.randomize_buf += 1

        # TODO tune pushing
        # self.common_step_counter += 1
        # if self.common_step_counter % self.push_interval == 0:
        #     self.push_robots()

        # Turn off for testing
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def push_robots(self):
        self.root_states[..., self.bez_indices, 7:9] = torch_rand_float(-0.2, 0.2, (self.num_envs, 2),
                                                                        device=self.device)  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def imu(self):
        imu_val, self.prev_lin_vel = compute_imu(
            # tensors
            self.root_orient_bez,
            self.root_vel_bez,
            self.root_ang_bez,
            self.prev_lin_vel,
            self.gravity_vec,
            self.inv_start_rot,
            # floats
            self.imu_max_lin_acc,
            self.imu_max_ang_vel,
            self._IMU_LIN_STDDEV,
            self._IMU_ANG_STDDEV,
            self.dt,
            # int
            self.num_envs,
            self.device
        )
        return imu_val

    def off_orn(self):
        vec = compute_off_orn(
            # tensors
            self.root_pos_bez,
            self.root_orient_bez,
            self.goal
        )
        return vec

    def feet_sensors_cleats(self):
        """
        TODO debug weird behavior
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
        forces = torch.tensor([[-1.] * self.feet_dim], device=self.device).repeat((self.num_envs, 1))  # nx8
        ones = torch.ones_like(forces)

        location = compute_feet_sensors_cleats(
            # tensors
            self.left_contact_forces,
            self.right_contact_forces,
            forces,
            ones
        )

        # print("self.left_foot_contact_forces: ", self.left_foot_contact_forces)
        # print("self.right_foot_contact_forces: ", self.right_foot_contact_forces)
        # print('feet: ', location)

        return location

    def feet_sensors_no_cleats(self):
        """
        TODO debug weird behavior
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
        forces = torch.tensor([[-1.] * 4], device=self.device).repeat((self.num_envs, 1))

        # General tensors
        ones = torch.ones(1, device=self.device)
        zeros = torch.zeros(1, device=self.device)
        zero = torch.zeros(3, device=self.device)

        # Possible sensor configuration
        forces_case_1 = torch.tensor([1., -1., -1., -1.],
                                     device=self.device)
        forces_case_2 = torch.tensor([-1., -1., 1., -1.],
                                     device=self.device)
        forces_case_3 = torch.tensor([1., -1., 1., -1.],
                                     device=self.device)
        forces_case_5 = torch.tensor([-1., 1., -1., -1.],
                                     device=self.device)
        forces_case_6 = torch.tensor([-1., -1., -1., 1.],
                                     device=self.device)
        forces_case_7 = torch.tensor([-1., 1., -1., 1.],
                                     device=self.device)
        forces_case_9 = torch.tensor([1., 1., -1., -1.],
                                     device=self.device)
        forces_case_10 = torch.tensor([-1., -1., 1., 1.],
                                      device=self.device)
        forces_case_11 = torch.tensor([1., 1., 1., 1.],
                                      device=self.device)
        forces_case_12 = torch.tensor([-1., -1., -1., -1.], device=self.device)

        left_forces = compute_feet_sensors_no_cleats(
            # tensors
            self.left_foot_contact_forces,
            forces,
            ones,
            zeros,
            zero,
            forces_case_1,
            forces_case_2,
            forces_case_3,
            forces_case_5,
            forces_case_6,
            forces_case_7,
            forces_case_9,
            forces_case_10,
            forces_case_11,
            forces_case_12
        )

        right_forces = compute_feet_sensors_no_cleats(
            # tensors
            self.right_foot_contact_forces,
            forces,
            ones,
            zeros,
            zero,
            forces_case_1,
            forces_case_2,
            forces_case_3,
            forces_case_5,
            forces_case_6,
            forces_case_7,
            forces_case_9,
            forces_case_10,
            forces_case_11,
            forces_case_12
        )

        location = torch.cat((left_forces, right_forces), 1)

        # print("self.left_foot_contact_forces: ", self.left_foot_contact_forces)
        # print("self.right_foot_contact_forces: ", self.right_foot_contact_forces)
        # print('feet: ', location)

        return location

    def compute_reward(self, actions):

        if self.debug_rewards:
            roll, pitch, yaw, up_proj, root_pos_bez, \
            ball_to_goal_angle, ball_init_to_goal_angle, goal_angle_diff, \
            distance_traveled_norm, distance_to_goal_x, distance_to_goal_y, \
            distance_to_goal_norm, distance_kicked_norm, current_time, \
            avg_kick, current_velocity_norm, \
            ball_vel_height_reward, ball_velocity_forward_reward_scaled_after, \
            velocity_forward_reward_scaled, vel_reward_scaled, pos_reward_scaled, \
            dof_vel_reward_scaled, distance_to_height_scaled, distance_to_height_orig, \
            goal_angle_diff_scaled, ground_feet_scaled, feet = compute_bez_reward_calculations(
                # tensors
                self.dof_pos_bez,
                self.dof_vel_bez,
                self.default_dof_pos,
                self.root_vel_bez,
                self.root_ang_bez,
                self.root_pos_bez,
                self.root_orient_bez,
                self.up_vec,
                self.root_pos_ball,
                self.root_vel_ball,
                self.goal,
                self.ball_init,
                self.bez_init_xy,
                self.progress_buf,
                self.feet,
                # self.left_arm_contact_forces,
                # self.right_arm_contact_forces,
                # int
                self.num_envs,
                # float
                self.dt,

            )
            if current_time > 0:
                temp = distance_kicked_norm / current_time
                self.max_kick_velocity = max(self.max_kick_velocity, temp)

                #

                step = int(current_time / self.dt + 0.5)
                kick_velocity = np.array(self.kick_vel)
                # import matplotlib.animation as animation
                #
                # ani = animation.FuncAnimation(self.fig, self.animate, fargs=(step, temp), interval=1000)

                self.kick_vel.append(temp)
                self.up_proj.append(up_proj)
                self.goal_angle_diff.append(goal_angle_diff)
                self.distance_kicked_norm.append(distance_kicked_norm)
                self.vel_reward_scaled.append(vel_reward_scaled)
                self.pos_reward_scaled.append(pos_reward_scaled)
                self.time.append(step)

                self.ax[0, 0].plot(self.time, self.kick_vel)
                self.ax[0, 1].plot(self.time, self.distance_kicked_norm)
                self.ax[0, 2].plot(self.time, self.goal_angle_diff)
                self.ax[1, 0].plot(self.time, self.up_proj)
                self.ax[1, 1].plot(self.time, self.vel_reward_scaled)
                self.ax[1, 2].plot(self.time, self.pos_reward_scaled)

                self.ax[0, 0].set_title('Kick Velocity')
                self.ax[0, 0].set_xlabel('time (t)')
                self.ax[0, 0].set_ylabel('Kick Velocity')

                self.ax[1, 0].set_title('up_proj')
                self.ax[1, 0].set_xlabel('time (t)')
                self.ax[1, 0].set_ylabel('up_proj')

                self.ax[0, 2].set_title('goal_angle_diff')
                self.ax[0, 2].set_xlabel('time (t)')
                self.ax[0, 2].set_ylabel('goal_angle_diff')

                self.ax[0, 1].set_title('distance_kicked_norm')
                self.ax[0, 1].set_xlabel('time (t)')
                self.ax[0, 1].set_ylabel('distance_kicked_norm')

                self.ax[1, 1].set_title('vel_reward_scaled')
                self.ax[1, 1].set_xlabel('time (t)')
                self.ax[1, 1].set_ylabel('vel_reward_scaled')

                self.ax[1, 2].set_title('pos_reward_scaled')
                self.ax[1, 2].set_xlabel('time (t)')
                self.ax[1, 2].set_ylabel('pos_reward_scaled')
                self.fig.canvas.manager.show()
                # print(self.kick_vel)
                # print(self.time)
                # Foot Step ratio

                # Format plot

                # self.fig.canvas.draw()
                plt.draw()
                plt.pause(0.000000000001)

                # self.fig.canvas.flush_events()
                # print("times: ", times.shape)
                # print("kick_velocity: ", kick_velocity.shape)

            # sys.stdout.write('\r' + "roll: " + str(roll) +
            #                  "      pitch: " + str(pitch) +
            #                  "      up_vec: " + str(up_proj) +
            #                  "      root_pos_bez: " + str(root_pos_bez) +
            #                  "      ball_to_goal_angle: " + str(ball_to_goal_angle) +
            #                  "      ball_init_to_goal_angle: " + str(ball_init_to_goal_angle) +
            #                  "      goal_angle_diff: " + str(goal_angle_diff) +
            #                  "      distance_traveled_norm: " + str(distance_traveled_norm) +
            #                  "      distance_to_goal_x: " + str(distance_to_goal_x) +
            #                  "      distance_to_goal_y: " + str(distance_to_goal_y) +
            #                  "      distance_to_goal_norm: " + str(distance_to_goal_norm) +
            #                  "      distance_kicked_norm: " + str(distance_kicked_norm) +
            #                  "      Time elapsed:: " + str(current_time)  +
            #                  "      max_kick_velocity: " + str(self.max_kick_velocity) +
            #                  "      average_velocity_norm: " + str(current_time) +
            #                  "      current_velocity_norm: " + str(current_velocity_norm) +
            #                  "      ball_vel_height_reward: " + str(ball_vel_height_reward) +
            #                  "      ball_velocity_forward_reward_scaled: " + str(
            #     ball_velocity_forward_reward_scaled_after) +
            #                  "      velocity_forward_reward_scaled: " + str(velocity_forward_reward_scaled) +
            #                  "      vel_reward_scaled: " + str(vel_reward_scaled) +
            #                  "      pos_reward_scaled: " + str(pos_reward_scaled) +
            #                  "      dof_vel_reward_scaled: " + str(dof_vel_reward_scaled) +
            #                  "      distance_to_height_scaled: " + str(distance_to_height_scaled) +
            #                  "      goal_angle_diff_scaled: " + str(goal_angle_diff_scaled) +
            #                  "      ground_feet_scaled: " + str(ground_feet_scaled) +
            #                  "      feet: " + str(int(feet[0,0])) +
            #                  ", " + str(int(feet[0, 1])) +
            #                  ", " + str(int(feet[0, 2])) +
            #                  ", " + str(int(feet[0, 3])) +
            #                  ", " + str(int(feet[0, 4])) +
            #                  ", " + str(int(feet[0, 5])) +
            #                  ", " + str(int(feet[0, 6])) +
            #                  ", " + str(int(feet[0, 7]))
            #
            #                  )
            #
            # sys.stdout.flush()

        self.rew_buf[:], self.reset_buf[:] = compute_bez_reward(
            # tensors
            self.dof_pos_bez,
            self.dof_vel_bez,
            self.default_dof_pos,
            self.root_vel_bez,
            self.root_ang_bez,
            self.root_pos_bez,
            self.root_orient_bez,
            self.up_vec,
            self.goal,
            self.bez_init_xy,
            self.reset_buf,
            self.progress_buf,
            self.feet,
            # self.left_arm_contact_forces,
            # self.right_arm_contact_forces,
            # int
            self.max_episode_length,
            self.num_envs,
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # print('left_arm_contact_forces: ', self.left_arm_contact_forces, torch.linalg.norm(
        # self.left_arm_contact_forces.T, dim=0).T) print('right_arm_contact_forces: ',
        # self.right_arm_contact_forces, torch.linalg.norm(self.right_arm_contact_forces.T, dim=0).T)
        # up_vec = get_basis_vector(root_orient_bez[..., 0:4], up_vec).view(num_envs, 3)
        # up_proj = up_vec[:, 2]
        # imu = self.imu()
        # off_orn = self.off_orn()

        # self.feet = torch.tensor([[-1.] * self.feet_dim], device=self.device).repeat((self.num_envs, 1))
        # if self.cleats:
        #     self.feet = self.feet_sensors_cleats()
        # else:
        #     self.feet = self.feet_sensors_no_cleats()

        # print('imu: ', imu.cpu().numpy())
        # print('feet: ', self.feet)

        # noise
        # temp_dof_pos_bez = self.dof_pos_bez + torch.normal(0.0, self._JOIN_ANGLE_STDDEV, size=(self.num_envs, self.dof_dim), device=self.device)
        # temp_dof_vel_bez = self.dof_vel_bez + torch.normal(0.0, self._JOIN_VELOCITY_STDDEV, size=(self.num_envs, self.dof_dim), device=self.device)

        self.obs_buf[:] = compute_bez_observations(
            # tensors
            self.dof_pos_bez,  # 18
            self.dof_vel_bez,  # 18
            self.root_orient_bez  # 4
            # imu,  # 6
            # off_orn,  # 2
            # self.feet,  # 8
            # self.ball_init  # 2
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # DOF randomization
        positions_offset = torch_rand_float(-0.15, 0.15, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos_bez[env_ids] = tensor_clamp(self.default_dof_pos[env_ids] + positions_offset,
                                                 self.dof_pos_limits_lower, self.dof_pos_limits_upper)
        self.dof_vel_bez[env_ids] = velocities

        self.kick_vel = []
        self.up_proj = []
        self.goal_angle_diff = []
        self.distance_kicked_norm = []
        self.vel_reward_scaled = []
        self.pos_reward_scaled = []
        self.time = []

        if self.debug_rewards:
            self.ax[0, 0].cla()
            self.ax[0, 1].cla()
            self.ax[0, 2].cla()
            self.ax[1, 0].cla()
            self.ax[1, 1].cla()
            self.ax[1, 2].cla()

        # goal pos randomization
        # goal_x = torch_rand_float(0.5, 1.5, (len(env_ids), 1), device=self.device)
        # goal_y = torch_rand_float(-0.5, 0.5, (len(env_ids), 1), device=self.device)
        #
        # self.goal[env_ids, 0] = goal_x[0]
        # self.goal[env_ids, 1] = goal_y[0]

        # ball pos randomization
        # ball_x = torch_rand_float(0.175, 0.2, (len(env_ids), 1), device=self.device)
        # ball_y = torch_rand_float(-0.05, 0.05, (len(env_ids), 1), device=self.device)
        # print("ball: ", ball_x, ball_y)
        # print("goal: ", goal_x, goal_y)
        # print(ball_indices)
        # print(env_ids)
        # self.initial_root_states[ball_indices, 0] = ball_x[0]
        # self.initial_root_states[ball_indices, 1] = ball_y[0]
        # self.ball_init[env_ids, 0] = ball_x[0]
        # self.ball_init[env_ids, 1] = ball_y[0]

        bez_indices = self.bez_indices[env_ids].to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(bez_indices), len(bez_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.default_dof_pos),
                                                        gymtorch.unwrap_tensor(bez_indices),
                                                        len(bez_indices))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(bez_indices),
                                              len(bez_indices))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))


@torch.jit.script
def compute_imu(
        # tensors
        root_orient_bez: Tensor,
        root_vel_bez: Tensor,
        root_ang_bez: Tensor,
        prev_lin_vel: Tensor,
        gravity_vec: Tensor,
        inv_start_rot: Tensor,
        # floats
        imu_max_lin_acc: float,
        imu_max_ang_vel: float,
        _IMU_LIN_STDDEV: float,
        _IMU_ANG_STDDEV: float,
        dt: float,
        # int
        num_envs: int,
        device: str

) -> Tuple[Tensor, Tensor]:
    torso_quat = quat_mul(root_orient_bez, inv_start_rot)

    vel_loc = quat_rotate_inverse(torso_quat, root_vel_bez)
    angvel_loc = quat_rotate_inverse(torso_quat, root_ang_bez)
    #
    # print("root_orient_bez: ", root_orient_bez)
    # print("inv_start_rot: ", inv_start_rot)
    # print("torso_quat: ", torso_quat)
    # print("vel_loc: ", vel_loc)
    # print("root_vel_bez: ", root_vel_bez)
    # print("angvel_loc: ", angvel_loc)
    # print("root_ang_bez: ", root_ang_bez)

    lin_acc = torch.sub(vel_loc, prev_lin_vel)
    lin_acc = torch.div(lin_acc, dt)
    lin_acc = torch.sub(lin_acc, gravity_vec)
    # print('here1: ', lin_acc)
    # # rot_mat = quaternion_to_matrix(root_orient_bez)
    # # lin_acc_transform = torch.matmul(rot_mat, lin_acc.reshape((num_envs, -1, 1))).reshape(
    # #     (num_envs, -1,))  # nx3x3 * n*3*1
    # print('here2: ', lin_acc_transform)
    # Adding noise
    lin_acc_noise = torch.normal(0.0, _IMU_LIN_STDDEV, size=(num_envs, 3), device=device)
    ang_vel_noise = torch.normal(0.0, _IMU_ANG_STDDEV, size=(num_envs, 3), device=device)

    lin_acc_transform = lin_acc + lin_acc_noise
    angvel_loc = angvel_loc + ang_vel_noise

    lin_acc_clipped = torch.clamp(lin_acc_transform, -imu_max_lin_acc, imu_max_lin_acc)
    ang_vel_clipped = torch.clamp(angvel_loc, -imu_max_ang_vel, imu_max_ang_vel)
    imu_val = torch.cat([lin_acc_clipped, ang_vel_clipped], 1)

    return imu_val, vel_loc


@torch.jit.script
def compute_off_orn(
        # tensors
        root_pos_bez: Tensor,
        root_orient_bez: Tensor,
        goal: Tensor

) -> Tensor:
    distance_to_goal = torch.sub(goal, root_pos_bez[..., 0:2])
    distance_to_goal_norm = torch.reshape(torch.linalg.norm(distance_to_goal, dim=1), (-1, 1))
    distance_unit_vec = torch.div(distance_to_goal, distance_to_goal_norm)

    x = root_orient_bez[..., 0]
    y = root_orient_bez[..., 1]
    z = root_orient_bez[..., 2]
    w = root_orient_bez[..., 3]

    roll, pitch, yaw = get_euler_xyz(root_orient_bez[..., 0:4])
    cos = torch.cos(yaw)
    sin = torch.sin(yaw)
    d2_vect = torch.cat((cos.reshape((-1, 1)), sin.reshape((-1, 1))), dim=-1)
    cos = torch.sum(d2_vect * distance_unit_vec, dim=-1)
    distance_unit_vec_3d = torch.nn.functional.pad(input=distance_unit_vec, pad=(0, 1, 0, 0), mode='constant',
                                                   value=0.0)
    d2_vect_3d = torch.nn.functional.pad(input=d2_vect, pad=(0, 1, 0, 0), mode='constant',
                                         value=0.0)
    sin = torch.linalg.norm(torch.cross(distance_unit_vec_3d, d2_vect_3d, dim=1), dim=1)
    vec = torch.cat((sin.reshape((-1, 1)), -cos.reshape((-1, 1))), dim=1)

    return vec


# No cleats
@torch.jit.script
def compute_feet_sensors_no_cleats(
        # tensors
        foot_contact_forces: Tensor,
        forces: Tensor,
        ones: Tensor,
        zeros: Tensor,
        zero: Tensor,
        forces_case_1: Tensor,
        forces_case_2: Tensor,
        forces_case_3: Tensor,
        forces_case_5: Tensor,
        forces_case_6: Tensor,
        forces_case_7: Tensor,
        forces_case_9: Tensor,
        forces_case_10: Tensor,
        forces_case_11: Tensor,
        forces_case_12: Tensor,

) -> Tensor:
    # Filter noise
    foot_contact_forces[..., 0:3] = torch.where(
        torch.abs(foot_contact_forces[..., 0:3]) > 0.01,
        foot_contact_forces[..., 0:3],
        zero)

    # x sign
    x = torch.where(
        torch.abs(foot_contact_forces[..., 0]) > 0.0, ones, zeros)
    x = torch.where(foot_contact_forces[..., 0] == 0, 2.0 * torch.ones_like(ones), x)

    # y sign
    y = torch.where(
        torch.abs(foot_contact_forces[..., 1]) > 0.0, ones,
        3.0 * torch.ones_like(ones))
    y = torch.where(foot_contact_forces[..., 1] == 0, 3.0 * torch.ones_like(ones), y)

    # Determining sensors used
    sensor = torch.where(x == 1.0, zeros, 4.0 * torch.ones_like(ones))  # Is x positive
    sensor = torch.where(x == 2.0, 8.0 * torch.ones_like(ones), sensor)  # Is x zero
    case = torch.reshape(torch.add(y, sensor), (-1, 1))

    forces = torch.where(case == 1.0,
                         forces_case_1,
                         forces)  # Is sensor +,+
    forces = torch.where(case == 2.0,
                         forces_case_2,
                         forces)  # Is sensor +,-
    forces = torch.where(case == 3.0,
                         forces_case_3,
                         forces)  # Is sensor +,0
    forces = torch.where(case == 5.0,
                         forces_case_5,
                         forces)  # Is sensor -,+
    forces = torch.where(case == 6.0,
                         forces_case_6,
                         forces)  # Is sensor -,-
    forces = torch.where(case == 7.0,
                         forces_case_7,
                         forces)  # Is sensor -,0
    forces = torch.where(case == 9.0,
                         forces_case_9,
                         forces)  # Is sensor 0,+
    forces = torch.where(case == 10.0,
                         forces_case_10,
                         forces)  # Is sensor 0,-
    forces = torch.where(case == 11.0,
                         forces_case_11,
                         forces)  # Is sensor 0,0

    forces = torch.where(torch.reshape(foot_contact_forces[..., 2], (-1, 1)) < 1,
                         forces_case_12,
                         forces)  # Is negative force

    return forces


# Cleats
@torch.jit.script
def compute_feet_sensors_cleats(
        # tensors
        left_contact_forces: Tensor,
        right_contact_forces: Tensor,
        forces: Tensor,
        ones: Tensor
) -> Tensor:
    # original
    left_pts = torch.linalg.norm(left_contact_forces.T, dim=0).T  # nx4
    right_pts = torch.linalg.norm(right_contact_forces.T, dim=0).T  # nx4

    # Barely faster
    # left_pts = left_contact_forces[..., 2]
    # right_pts = right_contact_forces[..., 2]

    pts = torch.cat((left_pts, right_pts), 1)  # nx8
    location = torch.where(pts > 1.0, ones, forces)

    # print("self.left_pts: ", left_pts)
    # print("self.left_contact_forces: ", left_contact_forces)
    # print("self.right_pts: ", right_pts)
    # print("self.right_contact_forces: ", right_contact_forces)
    # print('feet: ', location)

    return location


@torch.jit.script
def compute_bez_reward_calculations(
        # tensors
        dof_pos_bez: Tensor,
        dof_vel_bez: Tensor,
        default_dof_pos: Tensor,
        imu_lin_bez: Tensor,
        imu_ang_bez: Tensor,
        root_pos_bez: Tensor,
        root_orient_bez: Tensor,
        up_vec: Tensor,
        root_pos_ball: Tensor,
        root_vel_ball: Tensor,
        goal: Tensor,
        ball_init: Tensor,
        bez_init_xy: Tensor,
        progress_buf: Tensor,
        feet: Tensor,
        # left_contact_forces: Tensor,
        # right_contact_forces: Tensor,
        num_envs: int,
        dt: float

) -> Tuple[
    float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, float, Any]:  # (reward, reset)
    # Calculate Velocity direction field
    distance_to_ball = torch.sub(root_pos_ball[..., 0:2], root_pos_bez[..., 0:2])
    distance_to_ball_norm = torch.reshape(torch.linalg.norm(distance_to_ball, dim=1), (-1, 1))
    bez_to_ball_unit_vec = torch.div(distance_to_ball, distance_to_ball_norm)
    velocity_forward_reward = torch.sum(torch.mul(bez_to_ball_unit_vec, imu_lin_bez[..., 0:2]), dim=1)

    distance_to_goal = torch.sub(goal, root_pos_ball[..., 0:2])
    distance_to_goal_norm = torch.reshape(torch.linalg.norm(distance_to_goal, dim=1), (-1, 1))
    ball_to_goal_unit_vec = torch.div(distance_to_goal, distance_to_goal_norm)
    ball_velocity_forward_reward = torch.sum(torch.mul(ball_to_goal_unit_vec, root_vel_ball[..., 0:2]), dim=1)

    # Testing
    # ball_init[..., 0] = 0.175
    # ball_init[..., 0] = 0.0

    distance_to_goal_init = torch.sub(goal, ball_init)  # nx2
    distance_to_goal_init_norm = torch.reshape(torch.linalg.norm(distance_to_goal_init, dim=1), (-1, 1))
    ball_init_to_goal_unit_vec = torch.div(distance_to_goal_init, distance_to_goal_init_norm)  # 2xn / nx1 = nx2

    ball_to_goal_angle = torch.atan2(ball_to_goal_unit_vec[..., 1], ball_to_goal_unit_vec[..., 0])
    ball_init_to_goal_angle = torch.atan2(ball_init_to_goal_unit_vec[..., 1], ball_init_to_goal_unit_vec[..., 0])
    goal_angle_diff = torch.abs((ball_init_to_goal_angle - ball_to_goal_angle))
    goal_angle_diff = torch.reshape(goal_angle_diff, (-1))

    up_vec = get_basis_vector(root_orient_bez[..., 0:4], up_vec).view(num_envs, 3)
    up_proj = up_vec[:, 2]
    roll, pitch, yaw = get_euler_xyz(root_orient_bez[..., 0:4])

    DESIRED_HEIGHT = 0.325  # Height of ready position

    # reward

    # old
    vel_bez = torch.cat((imu_lin_bez, imu_ang_bez), dim=1)
    vel_reward = torch.linalg.norm(vel_bez, dim=1)
    vel_reward_scaled = torch.mul(vel_reward, 0.05)
    # new
    # vel_lin_reward = torch.mul(torch.linalg.norm(imu_lin_bez, dim=1), 0.05)
    # vel_ang_reward = torch.mul(torch.linalg.norm(imu_ang_bez, dim=1), 0.05)
    # vel_reward = torch.add(vel_lin_reward, vel_ang_reward)

    pos_reward = torch.linalg.norm(default_dof_pos - dof_pos_bez, dim=1)
    pos_reward_scaled = torch.mul(pos_reward, 0.05)
    distance_to_height_orig = torch.abs(0.325 - root_pos_bez[..., 2])
    distance_to_height = torch.abs(DESIRED_HEIGHT - root_pos_bez[..., 2])
    # distance_to_height = torch.abs(DESIRED_HEIGHT - up_proj)
    distance_to_height_scaled = torch.mul(distance_to_height, 1)
    distance_kicked_norm = torch.linalg.norm(torch.sub(root_pos_ball[..., 0:2], ball_init), dim=1)
    distance_traveled_norm = torch.reshape(torch.linalg.norm(torch.sub(root_pos_bez[..., 0:2], bez_init_xy), dim=1),
                                           (-1))
    goal_angle_diff_scaled = torch.mul(goal_angle_diff, 0.1)

    # Feet reward
    ground_feet = torch.sum(feet, dim=1)
    ground_feet_scaled = torch.mul(ground_feet, 0.01)  # not necessary
    # ground_feet_scaled = torch.sub(ground_feet_scaled, 0.004)

    # DOF Velocity reward
    dof_vel_reward = torch.linalg.norm(dof_vel_bez, dim=1)
    dof_vel_reward_scaled = torch.mul(dof_vel_reward, 0.001)  # doesnt work

    #  0.1 * ball_velocity_forward_reward - ((distance_to_height + (0.05 * vel_reward + 0.05 * pos_reward)) - 0.01 * ground_feet)
    #  0.1 * ball_velocity_forward_reward - distance_to_height - 0.05 * vel_reward - 0.05 * pos_reward
    #  0.1 * ball_velocity_forward_reward - distance_to_height - 0.05 * vel_reward - 0.05 * pos_reward + 0.01 * ground_feet
    #  0.1 * ball_velocity_forward_reward - distance_to_height - 0.05 * vel_reward - 0.05 * pos_reward - 0.01 * goal_angle_diff
    vel_pos_reward = torch.add(vel_reward_scaled, pos_reward_scaled)
    height_vel_pos_reward = torch.add(distance_to_height_scaled, vel_pos_reward)
    # height_vel_pos_reward = torch.sub(height_vel_pos_reward, ground_feet_scaled)
    # height_vel_pos_reward = torch.add(height_vel_pos_reward, goal_angle_diff_scaled)
    # height_vel_pos_reward = torch.add(height_vel_pos_reward, dof_vel_reward_scaled)
    # height_pos_reward = torch.add(distance_to_height, pos_reward)
    # height_pos_reward_scaled = torch.mul(height_pos_reward, 1)
    ball_velocity_forward_reward_scaled_after = torch.mul(ball_velocity_forward_reward, 0.1)
    ball_velocity_forward_reward_scaled_before = torch.mul(ball_velocity_forward_reward, 0.1)
    ball_height_vel_pos_reward = torch.sub(ball_velocity_forward_reward_scaled_after, height_vel_pos_reward)

    # 0.1 * ball_velocity_forward_reward + 0.05 * velocity_forward_reward - distance_to_height
    velocity_forward_reward_scaled = torch.mul(velocity_forward_reward, 0.05)
    vel_height_reward = torch.sub(velocity_forward_reward_scaled, distance_to_height_scaled)
    # height_goal_reward = torch.add(distance_to_height_scaled, goal_angle_diff_scaled)
    # vel_height_reward = torch.sub(velocity_forward_reward_scaled, height_goal_reward)
    ball_vel_height_reward = torch.add(ball_velocity_forward_reward_scaled_before, vel_height_reward)

    current_velocity_norm = torch.reshape(torch.linalg.norm(root_vel_ball[..., 0:2], dim=1), (-1, 1))
    distance_to_goal_x = distance_to_goal[..., 0].float()
    distance_to_goal_y = distance_to_goal[..., 1].float()
    current_time = float((progress_buf)[0] * dt)
    avg_kick = float(distance_kicked_norm[0]) / current_time

    return float(normalize_angle(roll[0])), float(normalize_angle(pitch[0])), float(normalize_angle(yaw[0])), float(
        up_proj[0]), float(root_pos_bez[0, 2]), \
           float(ball_to_goal_angle[0]), float(ball_init_to_goal_angle[0]), float(goal_angle_diff[0]), \
           float(distance_traveled_norm[0]), float(distance_to_goal_x[0]), float(distance_to_goal_y[0]), \
           float(distance_to_goal_norm[0]), float(distance_kicked_norm[0]), current_time, \
           avg_kick, float(current_velocity_norm[0]), \
           float(ball_vel_height_reward[0]), float(ball_velocity_forward_reward_scaled_after[0]), \
           float(velocity_forward_reward_scaled[0]), float(vel_reward_scaled[0]), float(pos_reward_scaled[0]), \
           float(dof_vel_reward_scaled[0]), float(distance_to_height_scaled[0]), float(distance_to_height_orig[0]), \
           float(goal_angle_diff_scaled[0]), float(ground_feet_scaled[0]), feet.float()


@torch.jit.script
def compute_bez_reward(
        # tensors
        dof_pos_bez: Tensor,
        dof_vel_bez: Tensor,
        default_dof_pos: Tensor,
        imu_lin_bez: Tensor,
        imu_ang_bez: Tensor,
        root_pos_bez: Tensor,
        root_orient_bez: Tensor,
        up_vec: Tensor,
        goal: Tensor,
        bez_init_xy: Tensor,
        reset_buf: Tensor,
        progress_buf: Tensor,
        feet: Tensor,
        # left_contact_forces: Tensor,
        # right_contact_forces: Tensor,
        max_episode_length: int,
        num_envs: int

) -> Tuple[Tensor, Tensor]:  # (reward, reset)
    # Calculate Velocity direction field

    # Testing
    # ball_init[..., 0] = 0.175
    # ball_init[..., 0] = 0.0

    up_vec = get_basis_vector(root_orient_bez[..., 0:4], up_vec).view(num_envs, 3)
    up_proj = up_vec[:, 2]
    roll, pitch, yaw = get_euler_xyz(root_orient_bez[..., 0:4])

    DESIRED_HEIGHT = 0.325  # Height of ready position

    # reward

    # old
    vel_bez = torch.cat((imu_lin_bez, imu_ang_bez), dim=1)
    vel_reward = torch.linalg.norm(vel_bez, dim=1)
    vel_reward_scaled = torch.mul(vel_reward, 0.05)
    # new
    # vel_lin_reward = torch.mul(torch.linalg.norm(imu_lin_bez, dim=1), 0.05)
    # vel_ang_reward = torch.mul(torch.linalg.norm(imu_ang_bez, dim=1), 0.05)
    # vel_reward = torch.add(vel_lin_reward, vel_ang_reward)

    pos_reward = torch.linalg.norm(default_dof_pos - dof_pos_bez, dim=1)
    pos_reward_scaled = torch.mul(pos_reward, 0.05)
    distance_to_height = torch.abs(DESIRED_HEIGHT - root_pos_bez[..., 2])
    # distance_to_height = torch.abs(DESIRED_HEIGHT - up_proj)
    distance_to_height_scaled = torch.mul(distance_to_height, 1)

    # Feet reward
    ground_feet = torch.sum(feet, dim=1)
    ground_feet_scaled = torch.mul(ground_feet, 0.01)  # not necessary
    # ground_feet_scaled = torch.sub(ground_feet_scaled, 0.004)

    # DOF Velocity reward
    dof_vel_reward = torch.linalg.norm(dof_vel_bez, dim=1)
    dof_vel_reward_scaled = torch.mul(dof_vel_reward, 0.001)  # doesnt work

    #  0.1 * ball_velocity_forward_reward - ((distance_to_height + (0.05 * vel_reward + 0.05 * pos_reward)) - 0.01 * ground_feet)
    #   - distance_to_height - 0.05 * vel_reward - 0.05 * pos_reward
    #  0.1 * ball_velocity_forward_reward - distance_to_height - 0.05 * vel_reward - 0.05 * pos_reward + 0.01 * ground_feet
    #  0.1 * ball_velocity_forward_reward - distance_to_height - 0.05 * vel_reward - 0.05 * pos_reward - 0.01 * goal_angle_diff
    vel_pos_reward = torch.add(vel_reward_scaled, pos_reward_scaled)
    height_vel_pos_reward = torch.add(distance_to_height_scaled, vel_pos_reward)
    # height_vel_pos_reward = torch.sub(height_vel_pos_reward, ground_feet_scaled)
    # height_vel_pos_reward = torch.add(height_vel_pos_reward, goal_angle_diff_scaled)
    # height_vel_pos_reward = torch.add(height_vel_pos_reward, dof_vel_reward_scaled)
    # height_pos_reward = torch.add(distance_to_height, pos_reward)
    # height_pos_reward_scaled = torch.mul(height_pos_reward, 1)

    reward = -1.0 * vel_pos_reward

    # Reset
    ones = torch.ones_like(reset_buf)
    termination_scale = -100.0

    # Fall
    reset = torch.where(up_proj < 0.93, ones, reset_buf)
    reward = torch.where(up_proj < 0.93, torch.ones_like(reward) * termination_scale, reward)  # -100
    # pitch_angle = torch.abs(normalize_angle(pitch))
    # reset = torch.where(pitch_angle > 1, ones, reset_buf)
    # reward = torch.where(pitch_angle > 1, torch.ones_like(reward) * termination_scale, reward)
    # reset = torch.where(up_proj < 0.7, ones, reset_buf)
    # reward = torch.where(up_proj < 0.7, torch.ones_like(reward) * termination_scale, reward)

    # Bez: Out of Bound
    distance_traveled_norm = torch.reshape(torch.linalg.norm(torch.sub(root_pos_bez[..., 0:2], bez_init_xy), dim=1),
                                           (-1))
    reset = torch.where(
        distance_traveled_norm > 0.2,
        ones,
        reset)
    reward = torch.where(
        distance_traveled_norm > 0.2,
        torch.ones_like(reward) * termination_scale,
        reward)

    # Horizon Ended
    reset = torch.where(progress_buf >= max_episode_length, ones, reset)

    reward = torch.where(progress_buf >= max_episode_length, torch.zeros_like(reward),
                         reward)

    # reset = torch.ones_like(reset_buf)

    return reward, reset


@torch.jit.script
def compute_bez_observations(
        # tensors
        dof_pos_bez: Tensor,
        dof_vel_bez: Tensor,  # 18
        imu: Tensor


) -> Tensor:
    obs = torch.cat((dof_pos_bez,
                     dof_vel_bez,  # 18
                     imu

                     ), dim=-1)

    return obs
