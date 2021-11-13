#!/usr/bin/env python3

# utragym
from isaacgym import gymutil, gymapi
from utragym.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg, retrieve_cfg
from utragym.utils.parse_task import parse_task

from utragym.envs import BezEnv
# python
import torch
import os
import yaml
import unittest

class TestBezEnv(unittest.TestCase):
    """Test the Trifinger gym environment."""

    """
    Reset environment tests
    """
    def test_default_reset(self):
        with open(os.path.join(os.getcwd(), 'resources/config/bez.yaml'), 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)

        # parse arguments
        args = gymutil.parse_arguments(
            description="RL Policy",
            headless=True,
            custom_parameters=[{"name": "--test", "action": "store_true", "default": False,
                                "help": "Run trained policy, no training"}])
        # Getting sim params
        sim_params = parse_sim_params(args, cfg)

        # create environment
        env = BezEnv(cfg=cfg, sim_params = sim_params, physics_engine = args.physics_engine, device_type = "cuda", device_id = args.compute_device_id, headless = args.headless)
        # env.create_sim()
        print(env.envs)
        # check reset
        for step_num in range(30000):
            # reset every certain number of steps
            if step_num % 100 == 0:
                env.post_physics_step()
            # render the env
            env.render()


if __name__ == '__main__':
    unittest.main()
