"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym",
    "torch",
    "omegaconf",
    "termcolor",
    "hydra-core>=1.1",
    "rl-games==1.1.3", # TODO upgrade to 1.5.2
    "pyvirtualdisplay",
    "matplotlib",
]

# Installation operation
setup(
    name="Bez_IsaacGym",
    author="Jonathan Spraggett",
    version="0.1.0",
    description="Isaac Gym Reinforcement Learning Environments for humanoid robot Bez.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.6.*",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7"],
    zip_safe=False,
)

# EOF
