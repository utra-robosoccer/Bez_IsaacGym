"""
Submodule contarining all the environments and registers them.
"""

from .kick_env import KickEnv
from .cartpole import Cartpole

# Mappings from strings to environments
isaacgym_task_map = {
    "bez_kick": KickEnv,
    "Cartpole": Cartpole,

}