# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import time

import isaacgym
import torch
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from utils.reformat import omegaconf_to_dict, print_dict
from utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator

from utils.utils import set_np_formatting, set_seed

from rl_games.common import env_configurations, vecenv
# from rl_games.torch_runner import Runner
from utils.torch_runner import Runner
import yaml
from hydra import compose, initialize


## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)


class LaunchModel:
    def __init__(self, checkpoint="results/Bez_Kick/Normal/Bez_Kick_33.pth", num_envs=1, env=None):
        initialize(config_path="./cfg")
        self.cfg = compose(config_name="config")
        self.checkpoint = checkpoint
        self.num_envs = num_envs
        self.env = env
        self.runner = None
        self.player = None



    def load_config(self):
        # ensure checkpoints can be specified as relative paths
        self.cfg.checkpoint = self.checkpoint
        self.cfg.num_envs = self.num_envs

        if self.cfg.checkpoint:
            self.cfg.checkpoint = to_absolute_path(self.cfg.checkpoint)

        # cfg_dict = omegaconf_to_dict(self.cfg)
        # print_dict(cfg_dict)
        # set numpy formatting for printing only
        set_np_formatting()

        # sets seed. if seed is -1 will pick a random one
        self.cfg.seed = set_seed(self.cfg.seed, torch_deterministic=self.cfg.torch_deterministic)

        # `create_rlgpu_env` is environment construction function which is passed to RL Games and called internally.
        # We use the helper function here to specify the environment config.
        create_rlgpu_env = get_rlgames_env_creator(
            omegaconf_to_dict(self.cfg.task),
            self.cfg.task_name,
            self.cfg.sim_device,
            self.cfg.rl_device,
            self.cfg.graphics_device_id,
            self.cfg.headless,
            multi_gpu=self.cfg.multi_gpu,
        )
        # register the rl-games adapter to use inside the runner
        vecenv.register('RLGPU',
                        lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
        })

        rlg_config_dict = omegaconf_to_dict(self.cfg.train)

        # convert CLI arguments into dictionory
        # create runner and set the settings
        self.runner = Runner(RLGPUAlgoObserver(),env=self.env)
        self.runner.load(rlg_config_dict)
        self.runner.reset()

        # dump config dict
        experiment_dir = os.path.join('runs', self.cfg.train.params.config.name)
        os.makedirs(experiment_dir, exist_ok=True)
        with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(self.cfg))

    def run_model(self):
        self.player = self.runner.create_player()
        self.player.restore(self.runner.load_path)

        if self.env is None:
            env = self.player.env
        else:
            env = self.env

        n_games = self.player.games_num
        render = self.player.render_env
        n_game_life = self.player.n_game_life
        is_determenistic = self.player.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(env, "has_action_mask", None) is not None

        op_agent = getattr(env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.player.env.create_agent(self.player.env.config)
            # self.player.env.set_weights(range(8),self.player.get_weights())

        if has_masks_func:
            has_masks = env.has_action_mask()

        need_init_rnn = self.player.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.player.env_reset(env)
            batch_size = 1
            batch_size = self.player.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.player.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.player.max_steps):
                if has_masks:
                    masks = env.get_action_mask()
                    action = self.player.get_masked_action(
                        obses, masks, is_determenistic)
                else:
                    action = self.player.get_action(obses, is_determenistic)
                obses, r, done, info = self.player.env_step(env, action)
                cr += r
                steps += 1

                if render:
                    env.render()
                    # env.render(mode='human')
                    # time.sleep(0.082)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.player.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.player.is_rnn:
                        for s in self.player.states:
                            s[:, all_done_indices, :] = s[:,
                                                        all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.player.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count)

                    sum_game_res += game_res
                    if batch_size // self.player.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)



if __name__ == "__main__":
    obj = LaunchModel()
    obj.load_config()
    obj.run_model()
    # launch_rlg_hydra()


