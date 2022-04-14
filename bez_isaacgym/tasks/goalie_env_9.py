import os
import time
import math
import sys

from matplotlib import pyplot as plt

from isaacgym.torch_utils import *
from utils.torch_jit_utils import *
from .base.vec_task import VecTask
from isaacgym import gymtorch
from isaacgym import gymapi
import numpy as np

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


class GoalieEnv(VecTask):

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

        # ball base init state
        pos = self.cfg["env"]["ballInitState"]["pos"]
        rot = self.cfg["env"]["ballInitState"]["rot"]
        v_lin = self.cfg["env"]["ballInitState"]["vLinear"]
        v_ang = self.cfg["env"]["ballInitState"]["vAngular"]
        self.ball_init_state = pos + rot + v_lin + v_ang

        # model choice
        self.cleats = self.cfg["env"]["asset"]["cleats"]

        # debug
        self.debug_rewards = self.cfg["env"]["debug"]["rewards"]

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["readyJointAngles"]  # defaultJointAngles  readyJointAngles

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
        self.ball_dim = 4  # ball position
        self.goal_line_dim = 2

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
            "numObservations"] = self.dof_dim + self.dof_dim + self.imu_dim + self.feet_dim + self.goal_line_dim + self.ball_dim# 56
        self.cfg["env"]["numActions"] = self.dof_dim

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)

        # simulation parameters
        self.dt = self.sim_params.dt
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        # TODO Implement
        self.orn_limit = torch.tensor([1.] * self.orn_dim, device=self.device)
        self.feet_limit = torch.tensor([1.6] * self.feet_dim, device=self.device)
        self.ball_start_limit = torch.tensor([0.3] * self.ball_dim, device=self.device)

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

        self.bez_init_xy = torch.tensor(self.bez_init_state[0:2], device=self.device)
        self.ball_init = torch.zeros(self.num_envs, 2, device=self.device)
        self.ball_init_vel = torch.zeros(self.num_envs, 2, device=self.device)

        # new ball spawning
        self.ball_angleRange = self.cfg["env"]["ballInitState"]["angleRange"]
        self.ball_distanceRange = self.cfg["env"]["ballInitState"]["distanceRange"]
        self.ball_distanceVariance = self.cfg["env"]["ballInitState"]["distanceVariance"]
        self.ball_ForwardAnglevariance = self.cfg["env"]["ballInitState"]["ForwardAnglevariance"]
        self.ball_speed = self.cfg["env"]["ballInitState"]["speed"]
        self.ball_speedVariance = self.cfg["env"]["ballInitState"]["speedVariance"]
        self.ball_base_shift = self.cfg["env"]["ballInitState"]["pos_deviation"]
        self.left_ratio = self.cfg["env"]["ballInitState"]["left_ratio"]

        # update ball position
        self.ball_position_generation()

        self.initial_root_states = self.root_states.clone()
        initial_root_states_bez = \
            to_torch([self.bez_init_state], device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        initial_root_states_ball = \
            to_torch([self.ball_init_state], device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        initial_root_states_ball[:, 0:2] = self.ball_init
        initial_root_states_ball[:, 7:9] = self.ball_init_vel

        self.initial_root_states[:] = \
            torch.cat([initial_root_states_bez, initial_root_states_ball], dim=-1).view(-1,initial_root_states_bez.shape[-1])

        self.goal_line_position = \
            torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        # root poz bez
        self.dof_pos_bez = self.dof_state.view(self.num_envs, self.num_dof, -1)[..., 0]
        self.dof_vel_bez = self.dof_state.view(self.num_envs, self.num_dof, -1)[..., 1]

        self.root_pos_bez = self.root_states.view(self.num_envs, -1, 13)[..., 0, 0:3]
        # self.root_orient_bez = self.root_states.view(self.num_envs, -1, 13)[..., 0, 3:7]

        # Imu link
        self.root_orient_bez = self.rigid_body.view(self.num_envs, -1, 13)[..., 1, 3:7]
        self.root_vel_bez = self.rigid_body.view(self.num_envs, -1, 13)[..., 1, 7:10]
        self.root_ang_bez = self.rigid_body.view(self.num_envs, -1, 13)[..., 1, 10:13]

        self.root_pos_ball = self.root_states.view(self.num_envs, -1, 13)[..., 1, 0:3]
        self.root_orient_ball = self.root_states.view(self.num_envs, -1, 13)[..., 1, 3:7]
        self.root_vel_ball = self.root_states.view(self.num_envs, -1, 13)[..., 1, 7:10]

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
        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # initialize some data used later on
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.time = []
        self.kick_vel = []
        self.up_proj = []
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

    def ball_position_generation(self):
        # update ball randomness
        angles = torch.rand(self.num_envs, device=self.device) \
                 * (self.ball_angleRange * 2) + math.pi / 2 - self.ball_angleRange
        distance = torch.rand(self.num_envs, device=self.device) \
                   * (self.ball_distanceVariance * 2) + self.ball_distanceRange - self.ball_distanceVariance
        forwandAngle = torch.rand(self.num_envs, device=self.device) \
                       * (self.ball_ForwardAnglevariance * 2) - self.ball_ForwardAnglevariance
        ball_speed = torch.rand(self.num_envs, device=self.device) \
                     * (self.ball_speedVariance * 2) + self.ball_speed - self.ball_speedVariance

        self.ball_init[:, 0] = torch.sin(angles) * distance
        self.ball_init[:, 1] = torch.cos(angles) * distance

        ball_angle_wf = torch.atan2(-self.ball_init[:, 0], -self.ball_init[:, 1]) + forwandAngle
        self.ball_init_vel[:, 0] = torch.sin(ball_angle_wf) * distance
        self.ball_init_vel[:, 1] = torch.cos(ball_angle_wf) * distance
        ball_vel_norm = torch.linalg.norm(self.ball_init_vel, dim=1)

        self.ball_init_vel[:, 0] = torch.div(self.ball_init_vel[:, 0], ball_vel_norm) * ball_speed
        self.ball_init_vel[:, 1] = torch.div(self.ball_init_vel[:, 1], ball_vel_norm) * ball_speed

        # add base shift
        self.ball_init[:, 0] = self.ball_init[:, 0] + torch.ones(self.num_envs, device=self.device) * self.ball_base_shift[0]
        #self.ball_init[:, 1] = self.ball_init[:, 1] + self.ball_base_shift[1] * \
        #    (torch.ones(self.num_envs, device=self.device) - 2 * torch.flatten(torch.randint(0, 2, (self.num_envs, 1), device=self.device)))

        ball_offset = torch.ones(self.num_envs, device=self.device) * -1 # ball_right
        left_num = int(self.num_envs * self.left_ratio)
        ball_left = torch.ones(left_num, device=self.device)
        ball_offset[:left_num] = ball_left

        # shuffle
        r = torch.randperm(self.num_envs, device=self.device)
        ball_offset = ball_offset[r]

        self.ball_init[:, 1] = self.ball_init[:, 1] + ball_offset * self.ball_base_shift[1]


        """
        # initial ball randomness to pos and vel
        self.ball_init = torch.rand(self.num_envs, 2, device=self.device) + torch.tensor([self.ball_init_state[0:2]], device=self.device).repeat((self.num_envs, 1))
        self.ball_init_vel = torch.rand(self.num_envs, 2, device=self.device) + torch.tensor([self.ball_init_state[7:9]], device=self.device).repeat((self.num_envs, 1))
        """
        """
        self.ball_init = torch.rand(self.num_envs, 2, device=self.device) + \
                         torch.tensor([self.ball_init_state[0:2]], device=self.device).repeat((self.num_envs, 1))
        self.ball_init_vel = torch.rand(self.num_envs, 2, device=self.device) + torch.tensor(
            [self.ball_init_state[7:9]], device=self.device).repeat((self.num_envs, 1))
        """

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
        asset_file_ball = "bez/objects/ball.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file_ball = self.cfg["env"]["asset"].get("assetFileNameBall", asset_file_ball)

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

        asset_root_ball = os.path.dirname(asset_path_ball)
        asset_file_ball = os.path.basename(asset_path_ball)

        asset_options_ball = gymapi.AssetOptions()
        asset_options_ball.default_dof_drive_mode = 0
        asset_options_ball.armature = 0.0  # self.cfg["env"]["urdfAsset"]["armature"]
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
        self.dof_vel_limits_upper = []
        self.dof_vel_limits_lower = []

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, bez_handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_pos_limits_lower.append(dof_prop['upper'][j])
                self.dof_pos_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_pos_limits_lower.append(dof_prop['lower'][j])
                self.dof_pos_limits_upper.append(dof_prop['upper'][j])

            self.dof_vel_limits_upper.append([self.MX_28_velocity])
            self.dof_vel_limits_lower.append([-self.MX_28_velocity])

        self.dof_pos_limits_lower = to_torch(self.dof_pos_limits_lower, device=self.device)
        self.dof_pos_limits_upper = to_torch(self.dof_pos_limits_upper, device=self.device)
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

        # Turn off for testing
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

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
            self.dt,
            # int
            self.num_envs
        )
        return imu_val

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
            self.root_pos_ball,
            self.root_vel_ball,
            self.ball_init,
            self.ball_init_vel,
            self.bez_init_xy,
            self.reset_buf,
            self.progress_buf,
            self.feet,
            self.max_episode_length,
            self.num_envs,
            self.goal_line_position
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

        imu = self.imu()

        # self.feet = torch.tensor([[-1.] * self.feet_dim], device=self.device).repeat((self.num_envs, 1))
        if self.cleats:
            self.feet = self.feet_sensors_cleats()
        else:
            self.feet = self.feet_sensors_no_cleats()

        self.complute_goaline_position()
        #print(self.goal_line_position)

        self.obs_buf[:] = compute_bez_observations(
            # tensors
            self.dof_pos_bez,  # 18
            self.dof_vel_bez,  # 18
            imu,  # 6
            self.feet,  # 8
            self.goal_line_position,  # 2
            self.ball_init,  # 2
            self.ball_init_vel # 2
        )

    def complute_goaline_position(self):
        # time to goal line (x/-vx)
        self.goal_line_position[:, 0] = torch.div(self.ball_init[:, 0], (-1 * self.ball_init_vel[:, 0]))
        # position at goal line (x/-vx * vy + y)
        self.goal_line_position[:, 1] = self.goal_line_position[:, 0] * self.ball_init_vel[:, 1] + self.ball_init[:, 1]

    def reset_idx(self, env_ids):
        # update ball position
        self.ball_position_generation()

        self.initial_root_states = self.root_states.clone()

        initial_root_states_bez = \
            to_torch([self.bez_init_state], device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        initial_root_states_ball = \
            to_torch([self.ball_init_state], device=self.device, requires_grad=False).repeat((self.num_envs, 1))
        initial_root_states_ball[:, 0:2] = self.ball_init
        initial_root_states_ball[:, 7:9] = self.ball_init_vel

        self.initial_root_states[:] = \
            torch.cat([initial_root_states_bez, initial_root_states_ball], dim=-1).view(-1, initial_root_states_bez.shape[-1])

        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        ball_indices = (env_ids * 2) + 1
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

        bez_indices = self.bez_indices[env_ids].to(dtype=torch.int32)
        ball_indices = self.ball_indices[env_ids].to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(bez_indices), len(bez_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(ball_indices), len(ball_indices))

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
        dt: float,
        # int
        num_envs: int

) -> Tuple[Tensor, Tensor]:
    # torso_quat = quat_mul(root_orient_bez, inv_start_rot)
    #
    # vel_loc = quat_rotate_inverse(torso_quat, root_vel_bez)
    # angvel_loc = quat_rotate_inverse(torso_quat, root_ang_bez)
    #
    # print("root_orient_bez: ", root_orient_bez)
    # print("inv_start_rot: ", inv_start_rot)
    # print("torso_quat: ", torso_quat)
    # print("vel_loc: ", vel_loc)
    # print("root_vel_bez: ", root_vel_bez)
    # print("angvel_loc: ", angvel_loc)
    # print("root_ang_bez: ", root_ang_bez)

    lin_acc = torch.sub(root_vel_bez, prev_lin_vel)
    lin_acc = torch.div(lin_acc, dt)
    lin_acc = torch.sub(lin_acc, gravity_vec)

    rot_mat = quaternion_to_matrix(root_orient_bez)
    lin_acc_transform = torch.matmul(rot_mat, lin_acc.reshape((num_envs, -1, 1))).reshape(
        (num_envs, -1,))  # nx3x3 * n*3*1

    lin_acc_clipped = torch.clamp(lin_acc_transform, -imu_max_lin_acc, imu_max_lin_acc)
    ang_vel_clipped = torch.clamp(root_ang_bez, -imu_max_ang_vel, imu_max_ang_vel)
    imu_val = torch.cat([lin_acc_clipped, ang_vel_clipped], 1)

    return imu_val, root_vel_bez


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


# todo FIX
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
        root_pos_ball: Tensor,
        root_vel_ball: Tensor,
        ball_init: Tensor,
        ball_init_vel: Tensor,
        bez_init_xy: Tensor,
        reset_buf: Tensor,
        progress_buf: Tensor,
        feet: Tensor,
        max_episode_length: int,
        num_envs: int,
        goal_line_position: Tensor

) -> Tuple[Tensor, Tensor]:  # (reward, reset, feet_in air, feet_air_time, episode sums)

    distance_to_ball = torch.sub(root_pos_ball[..., 0:2], root_pos_bez[..., 0:2])  # nx2
    distance_to_ball_norm = torch.reshape(torch.linalg.norm(distance_to_ball, dim=1), (-1, 1))
    distance_unit_vec = torch.div(distance_to_ball, distance_to_ball_norm)  # 2xn / nx1 = nx2

    distance_to_ball_goalline = torch.abs(torch.sub(goal_line_position[..., 0], root_pos_bez[..., 1]))  # nx1
    roll_abs = torch.abs(imu_ang_bez[..., 1])
    #print(imu_ang_bez[..., 1])

    ball_velocity_forward_reward = torch.sum(distance_unit_vec * root_vel_ball[..., 0:2], dim=-1)
    ball_velocity_norm = torch.reshape(torch.linalg.norm(root_vel_ball[..., 0:2], dim=1), (-1, 1))

    goalline_to_ball = torch.sub(root_pos_ball[..., 0:2], bez_init_xy[..., 0:2])  # nx2
    goalline_to_ball_norm = torch.reshape(torch.linalg.norm(goalline_to_ball, dim=1), (-1, 1))
    goalline_unit_vec = torch.div(goalline_to_ball, goalline_to_ball_norm)  # 2xn / nx1 = nx2

    # reward
    ones = torch.ones_like(reset_buf)
    #reward = (ones - torch.flatten(torch.tanh(distance_to_ball_goalline)))) * 0.01
    #reward = torch.zeros_like(reset_buf)
    reward = -1*roll_abs*0.02
    # 1-tanh(robot_to_ball)

    #print(goal_line_position)
    """
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
    """

    # =====================================
    # if ball stopped:
    #     print('Ball Stopped')
    # =====================================

    reset = torch.where(
        ball_velocity_norm[..., 0] < 0.1,
        ones,
        reset_buf
    )
    reward = torch.where(
        ball_velocity_norm[..., 0] < 0.1,
        2*ones - torch.flatten(torch.tanh(distance_to_ball_goalline)),
        reward
    )

    # =====================================
    # Failed to stop the ball
    # if torch.linalg.norm(root_pos_ball[..., 0:2] - goal) > 3 * torch.linalg.norm(goal.to(dtype=torch.float32)):
    #     print('Out of Bound')
    # =====================================

    reset = torch.where(
        goalline_to_ball[..., 0] < 0,
        ones,
        reset
    )
    reward = torch.where(
        goalline_to_ball[..., 0] < 0,
        torch.ones_like(reward) * -2.0,
        reward
    )

    # =====================================
    # if progress_buf >= max_episode_length:
    #     print('Horizon Ended')
    # =====================================

    # Horizon Ended
    reset = torch.where(progress_buf >= max_episode_length, ones, reset)
    reward = torch.where(
        progress_buf >= max_episode_length,
        (ones - torch.flatten(torch.tanh(distance_to_ball_norm))) * 0.5 \
            + (ones - torch.flatten(torch.tanh(ball_velocity_norm))) * 0.5 + ones,
        reward
    )

    # print(reward)
    return reward, reset


@torch.jit.script
def compute_bez_observations(
        # tensors
        dof_pos_bez: Tensor,  # 18
        dof_vel_bez: Tensor,  # 18
        imu: Tensor,  # 6
        feet: Tensor,  # 8
        goal_line_position: Tensor,  # 2
        ball_init: Tensor,  # 2
        ball_init_vel: Tensor  # 2

) -> Tensor:
    obs = torch.cat((dof_pos_bez,  # 18
                     dof_vel_bez,  # 18
                     imu,  # 6
                     feet,  # 8
                     goal_line_position,  # 2
                     ball_init,  # 2
                     ball_init_vel  # 2
                     ), dim=-1)

    return obs
