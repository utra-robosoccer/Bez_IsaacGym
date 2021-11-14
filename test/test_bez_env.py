#!/usr/bin/env python3

# utragym
from isaacgym import gymutil, gymapi
from utragym.utils.config import parse_sim_params
from utragym.utils import *
from utragym.envs import KickEnv
# python
import torch
import os
import yaml
import unittest
import sys


class TestBezEnv(unittest.TestCase):
    """Test the Utra gym environment."""

    def setUp(self) -> None:
        # load parameters
        with open(os.path.join(os.getcwd(), '../resources/config/bez.yaml'), 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)

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
            # sets the task type: that is implementation language
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

        # Getting sim params
        sim_params = parse_sim_params(args, cfg)

        # create environment
        self.env = KickEnv(cfg=cfg, sim_params=sim_params, physics_engine=args.physics_engine, device_type="cuda",
                           device_id=args.compute_device_id, headless=args.headless)

        # Testing parameter
        self.sim_length = 10000
        self.reset_length = 10000

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
                    self.env.reset(env_ids)

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
                    self.env.reset(env_ids)
            else:
                action = torch.zeros(self.env.actions.size(), dtype=torch.float, device=self.env.device)
                self.env.step(action, True)

            # render the env
            self.env.render()

    """
    Random step environment test
    """

    def test_random_action_agent(self):

        # check reset
        for step_num in range(self.sim_length):

            # reset every certain number of steps
            if step_num % self.reset_length == 0:
                env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if len(env_ids) > 0:
                    self.env.reset(env_ids)
            else:
                action = 3 * torch.rand(self.env.actions.size(), dtype=torch.float, device=self.env.device)
                self.env.step(action, True)

            # render the env
            self.env.render()

    """
    Motor environment test
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
                    self.env.reset(env_ids)
            else:
                # action = -3 * torch.rand(self.env.actions.size(), dtype=torch.float, device=self.env.device)
                # animate the dofs
                speed = 2
                # action[0][current_dof] = speed
                print(current_dof)
                for i in range(self.env.num_envs):
                    if anim_state == ANIM_SEEK_LOWER:
                        action[i][current_dof] -= speed* self.env.dt
                        if action[i][current_dof] <= self.env.dof_limits_lower[current_dof]:
                            action[i][current_dof] = self.env.dof_limits_lower[current_dof]
                            anim_state = ANIM_SEEK_UPPER
                    elif anim_state == ANIM_SEEK_UPPER:
                        action[i][current_dof] += speed * self.env.dt
                        if action[i][current_dof] >= self.env.dof_limits_upper[current_dof]:
                            action[i][current_dof] = self.env.dof_limits_upper[current_dof]
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
                print("here")
                print(action)
                print(self.env.dof_pos)
                self.env.step(action, True)

            # render the env
            self.env.render()
if __name__ == '__main__':
    unittest.main()
