"""
Submodule contarining all the environments and registers them.
"""

from .kick_env import KickEnv
from .walk_env import WalkEnv
from .orient_env import OrientEnv

# Mappings from strings to environments
isaacgym_task_map = {
    "bez_kick": KickEnv,
    "bez_walk": WalkEnv,
    "bez_orient": OrientEnv,


}