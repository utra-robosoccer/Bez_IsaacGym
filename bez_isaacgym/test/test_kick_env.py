#!/usr/bin/env python3

from isaacgym import gymutil, gymapi
from resources.library.geometry.src.soccer_geometry.transformation import Transformation
from resources.library.pycontrol.src.soccer_pycontrol import soccerbot_controller
from resources.library.trajectories.src.soccer_trajectories import SoccerTrajectoryClass
from play import LaunchModel

from bez_isaacgym.tasks.kick_env import KickEnv
# python
import torch
import os
import yaml
import unittest
import sys
from random import randint


class TestBezEnv(unittest.TestCase):
    """Test the Utra gym environment."""

    def setUp(self) -> None:

        with open(os.path.join(os.getcwd(), 'cfg/task/bez_kick_test.yaml'), 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        # load parameters
        with open(os.path.join(os.getcwd(), 'cfg/train/bez_kickPPO.yaml'), 'r') as f:
            cfg_train = yaml.load(f, Loader=yaml.SafeLoader)
        # remove file name from system arguments
        sys.argv.pop()

        # parse arguments
        custom_parameters = [
            # sets verbosity of the run
            {"name": "--verbose", "action": "store_true", "default": False,
             "help": "Set verbosity of the environment (useful for debugging)."},
            # sets whether to train/test
            {"name": "--test", "action": "store_true", "default": False,
             "help": "Run trained policy, no training"},
            # sets whether to train further or test with provided NN weights
            {"name": "--resume", "type": int, "default": 0,
             "help": "Resume training or start testing from a checkpoint"},
            # sets whether to run GUI for visualization or not.
            {"name": "--headless", "action": "store_true", "default": False,
             "help": "Force display off at all times"},
            # sets the tasks type: that is implementation language
            {"name": "--task_type", "type": str, "default": "Python",
             "help": "Choose Python or C++"},
            # sets the device for the environment
            {"name": "--device", "type": str, "default": "GPU",
             "help": "Choose CPU or GPU device for running physics"},
            # sets the device for the RL agent
            {"name": "--ppo_device", "type": str, "default": "GPU",
             "help": "Choose CPU or GPU device for inferencing PPO network"}
        ]

        args = gymutil.parse_arguments(
            description="RL Policy",
            custom_parameters=custom_parameters
        )

        # create environment
        self.env = KickEnv(cfg=cfg, sim_device=args.sim_device,
                           graphics_device_id=args.graphics_device_id, headless=args.headless)

        # Testing parameter
        self.sim_length = 30000
        self.reset_length = 30000

    """
    Reset environment tests
    """

    def test_default_reset(self):

        # check reset
        for step_num in range(self.sim_length):

            # reset every certain number of steps
            if step_num % self.reset_length == 0:
                env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if len(env_ids) > 0:
                    self.env.reset_idx(env_ids)

            # render the env
            self.env.render()

    """
    Step environment test
    """

    def test_zero_action_agent(self):

        # check reset
        for step_num in range(self.sim_length):

            # reset every certain number of steps
            if step_num % self.reset_length == 0:
                env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if len(env_ids) > 0:
                    self.env.reset_idx(env_ids)

            else:
                action = torch.zeros(self.env.actions.size(), dtype=torch.float, device=self.env.device)
                self.env.step(action)

            # render the env
            self.env.render()

    """
    Random step environment test
    Better when fixBaseLink = True
    """

    def test_random_action_agent(self):

        # check reset
        for step_num in range(1, self.sim_length):

            # reset every certain number of steps
            if step_num % self.reset_length == 0:
                print(step_num, self.reset_length)
                env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if len(env_ids) > 0:
                    self.env.reset_idx(env_ids)
            else:
                action = randint(-3, 3) * torch.rand(self.env.actions.size(), dtype=torch.float, device=self.env.device)
                self.env.step(action)

            # render the env
            self.env.render()

    """
    Motor environment test
    Better when fixBaseLink = True
    """

    def test_motor_action_agent(self):
        # joint animation states
        ANIM_SEEK_LOWER = 1
        ANIM_SEEK_UPPER = 2
        ANIM_SEEK_DEFAULT = 3
        ANIM_FINISHED = 4

        # initialize animation state
        anim_state = ANIM_SEEK_LOWER
        current_dof = 0
        # check reset
        action = torch.zeros(self.env.actions.size(), dtype=torch.float, device=self.env.device)
        for step_num in range(self.sim_length):
            # reset every certain number of steps
            if step_num % self.reset_length == 0:

                env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if len(env_ids) > 0:
                    self.env.reset_idx(env_ids)
            else:
                # animate the dofs
                speed = 3
                for i in range(self.env.num_envs):
                    if anim_state == ANIM_SEEK_LOWER:
                        action[i][current_dof] -= speed * self.env.dt
                        if action[i][current_dof] <= self.env.dof_pos_limits_lower[current_dof]:
                            action[i][current_dof] = self.env.dof_pos_limits_lower[current_dof]
                            anim_state = ANIM_SEEK_UPPER
                    elif anim_state == ANIM_SEEK_UPPER:
                        action[i][current_dof] += speed * self.env.dt
                        if action[i][current_dof] >= self.env.dof_pos_limits_upper[current_dof]:
                            action[i][current_dof] = self.env.dof_pos_limits_upper[current_dof]
                            anim_state = ANIM_SEEK_DEFAULT
                    if anim_state == ANIM_SEEK_DEFAULT:
                        action[i][current_dof] -= speed * self.env.dt
                        if action[i][current_dof] <= self.env.default_dof_pos[i][current_dof]:
                            action[i][current_dof] = self.env.default_dof_pos[i][current_dof]
                            anim_state = ANIM_FINISHED
                    elif anim_state == ANIM_FINISHED:
                        action[i][current_dof] = self.env.default_dof_pos[i][current_dof]
                        current_dof = (current_dof + 1) % 18
                        anim_state = ANIM_SEEK_LOWER
                self.env.step(action)
            # render the env
            self.env.render()

    """
    Walk environment test
    """

    def test_walk_agent(self):

        self.walker = soccerbot_controller.SoccerbotController(self.env, 0)
        self.walker.ready()
        self.walker.wait(100)
        self.walker.setGoal(Transformation([2, 0, 0], [0, 0, 0, 1]))
        # check reset
        for step_num in range(self.sim_length):
            # self.walker.soccerbot.robot_path.show()
            self.walker.run()

            # render the env
            self.env.render()

    """
    Trajectory environment test
    """

    def test_trajectory_agent(self):
        trajectory_class = SoccerTrajectoryClass(self.env, 0)
        self.walker = soccerbot_controller.SoccerbotController(self.env, 0)
        # self.walker.ready()
        self.walker.wait(100)
        # check reset
        for step_num in range(self.sim_length):
            # self.walker.soccerbot.robot_path.show()

            trajectory_class.run_trajectory("rightkick")

            # render the env
            self.env.render()

    """
    Trained Model environment test
    """

    def test_model_agent(self):
        obj = LaunchModel(env=self.env)
        obj.load_config()
        obj.run_model()


if __name__ == '__main__':
    unittest.main()
