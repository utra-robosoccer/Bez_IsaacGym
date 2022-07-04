# Bez_IsaacGym
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
![Python](https://img.shields.io/badge/python-3.8-blue.svg)
![Ubuntu 20.04](https://img.shields.io/badge/ubuntu-20.04-orange.svg)
[![Documentation Status](https://readthedocs.org/projects/soccerbot/badge/?version=latest)](https://soccerbot.readthedocs.io/en/latest/?badge=latest)

### About this repository
This repository provides IsaacGym environment for the [Humanoid Robot Bez](http://utrahumanoid.ca/our-project/).

The project currently uses [RL-Games 1.13](https://github.com/Denys88/rl_games) for training agents.

This code is released under [LICENSE](LICENSE).

# Installation
### Pre-requisites
The code has been tested on Ubuntu 20.04 with Python 3.8. The minimum recommended NVIDIA driver
version for Linux is `460.32`.
### Install IsaacGym
Download the Isaac Gym Preview 3 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation.

Once Isaac Gym is installed, to install all its dependencies, run:
```bash
cd PATH_TO/isaacgym/python
pip install -e .
```
To verify the details of the installed package, run:
```bash
pip show isaacgym
```

### Install Bez_IsaacGym

To install `Bez_IsaacGym` package and all its dependencies, run:
```bash
git clone git@github.com:utra-robosoccer/Bez_IsaacGym.git
cd PATH_TO/Bez_IsaacGym
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .
```

# Running
To train or test you must be in the [`bez_isaacgym`](bez_isaacgym) folder, run this line:
```bash
cd PATH_TO/Bez_IsaacGym/bez_isaacgym
```
## Training

We use [Hydra](https://hydra.cc/docs/intro/) to keep configuration of runs simple. You can view the main arguments to the scripts by looking in the file [`bez_isaacgym/cfg/config.yaml`](bez_isaacgym/cfg/config.yaml).

You can also set the configuration parameters from terminal by doing `{config_variable_name}={value}`. The main ones to be aware of for are:

* **`task`** (string): environment name to use.
* **`num_envs`** (int): number of environment instances to run. Default to 4096.
* **`headless`** (bool): whether to run the simulator with/without GUI.
* **`checkpoint`** (string): To load from a checkpoint.
* **`test`** (bool): whether to train.

To train your first policy, run this line:
```bash
python train.py task=bez_kick 
```

## Inference and Loading Checkpoints
Checkpoints are saved in the folder `runs/EXPERIMENT_NAME/nn` where `EXPERIMENT_NAME` 
defaults to the task name, but can also be overridden via the `experiment` argument.

To load a trained checkpoint and continue training, use the `checkpoint` argument:

```bash
python train.py task=bez_kick checkpoint=result/Bez_Kick/nn/Bez_Kick.pth
```

To load a trained checkpoint and only perform inference (no training), pass `test=True` 
as an argument, along with the checkpoint name. To avoid rendering overhead, you may 
also want to run with fewer environments using `num_envs=64`:

```bash
python train.py task=bez_kick checkpoint=runs/Bez_Kick/nn/Bez_Kick.pth test=True num_envs=64
```

Note that If there are special characters such as `[` or `=` in the checkpoint names, 
you will need to escape them and put quotes around the string. For example,
`checkpoint="./runs/Bez_Kick/nn/last_Bez_Kickep\=501rew\[5981.31\].pth"`

## Test
There are testing programs for sample behaviors located in [`bez_isaacgym/test`](bez_isaacgym/test).

Clicking on the green button next to each function with launch the test
