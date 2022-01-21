"""
Submodule contarining all the environments and registers them.
"""

# from utragym.tasks.kick_env import KickEnv
# from utragym.tasks.cartpole import Cartpole
from .kick_env import KickEnv
from .cartpole import Cartpole

# Mappings from strings to environments
isaacgym_task_map = {
    "bez_kick": KickEnv,
    "Cartpole": Cartpole,

}