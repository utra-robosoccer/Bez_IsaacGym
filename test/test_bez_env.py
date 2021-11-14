#!/usr/bin/env python3

# utragym
from isaacgym import gymutil, gymapi


from utragym.utils import *
from utragym.envs import BezEnv
# python
import torch
import os
import yaml
import unittest
import sys

from utragym.utils.config import parse_sim_params


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
        self.env = BezEnv(cfg=cfg, sim_params=sim_params, physics_engine=args.physics_engine, device_type="cuda",
                     device_id=args.compute_device_id, headless=args.headless)



    """
    Reset environment tests
    """

    def test_default_reset(self):

        # check reset
        for step_num in range(30000):

            # reset every certain number of steps
            if step_num % 1000 == 0:
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
        for step_num in range(30000):

            # reset every certain number of steps
            if step_num % 1000 == 0:
                env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if len(env_ids) > 0:
                    self.env.reset(env_ids)
            else:
                action = torch.zeros(self.env.actions.size(), dtype=torch.float, device=self.env.device)
                self.env.step(action)

            # render the env
            self.env.render()

    """
    Random step environment test
    """

    def test_zero_action_agent(self):

        # check reset
        for step_num in range(30000):

            # reset every certain number of steps
            if step_num % 1000 == 0:
                env_ids = self.env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
                if len(env_ids) > 0:
                    self.env.reset(env_ids)
            else:
                action = 1 * torch.rand(self.env.actions.size(), dtype=torch.float, device=self.env.device)
                self.env.step(action)

            # render the env
            self.env.render()


if __name__ == '__main__':
    unittest.main()
