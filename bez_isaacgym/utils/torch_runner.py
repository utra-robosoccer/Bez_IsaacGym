import time

import numpy as np
import copy
import torch
import yaml

from rl_games import envs
from rl_games.common import object_factory
from rl_games.common import env_configurations
from rl_games.common import experiment
from rl_games.common import tr_helpers

from rl_games.algos_torch import network_builder
from rl_games.algos_torch import model_builder
from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import a2c_discrete
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch import sac_agent
from utils.players import PpoPlayerContinuous

class Runner:
    def __init__(self, algo_observer=None, env = None):
        self.algo_factory = object_factory.ObjectFactory()
        self.algo_factory.register_builder('a2c_continuous', lambda **kwargs: a2c_continuous.A2CAgent(**kwargs))
        self.algo_factory.register_builder('a2c_discrete', lambda **kwargs: a2c_discrete.DiscreteA2CAgent(**kwargs))
        self.algo_factory.register_builder('sac', lambda **kwargs: sac_agent.SACAgent(**kwargs))
        # self.algo_factory.register_builder('dqn', lambda **kwargs : dqnagent.DQNAgent(**kwargs))

        self.player_factory = object_factory.ObjectFactory()
        self.player_factory.register_builder('a2c_continuous', lambda **kwargs: PpoPlayerContinuous(**kwargs))
        # self.player_factory.register_builder('a2c_discrete', lambda **kwargs: players.PpoPlayerDiscrete(**kwargs))
        # self.player_factory.register_builder('sac', lambda **kwargs: players.SACPlayer(**kwargs))
        # self.player_factory.register_builder('dqn', lambda **kwargs : players.DQNPlayer(**kwargs))

        self.model_builder = model_builder.ModelBuilder()
        self.network_builder = network_builder.NetworkBuilder()

        self.algo_observer = algo_observer
        self.env = env

        torch.backends.cudnn.benchmark = True

    def reset(self):
        pass

    def load_config(self, params):
        self.seed = params.get('seed', None)

        self.algo_params = params['algo']
        self.algo_name = self.algo_params['name']
        self.load_check_point = params['load_checkpoint']
        self.exp_config = None

        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)

        if self.load_check_point:
            print('Found checkpoint')
            print(params['load_path'])
            self.load_path = params['load_path']

        self.model = self.model_builder.load(params)
        self.config = copy.deepcopy(params['config'])

        self.config['reward_shaper'] = tr_helpers.DefaultRewardsShaper(**self.config['reward_shaper'])
        self.config['network'] = self.model

        has_rnd_net = self.config.get('rnd_config', None) != None
        if has_rnd_net:
            print('Adding RND Network')
            network = self.model_builder.network_factory.create(params['config']['rnd_config']['network']['name'])
            network.load(params['config']['rnd_config']['network'])
            self.config['rnd_config']['network'] = network

        has_central_value_net = self.config.get('central_value_config', None) != None
        if has_central_value_net:
            print('Adding Central Value Network')
            network = self.model_builder.network_factory.create(
                params['config']['central_value_config']['network']['name'])
            network.load(params['config']['central_value_config']['network'])
            self.config['central_value_config']['network'] = network

    def load(self, yaml_conf):
        self.default_config = yaml_conf['params']
        self.load_config(copy.deepcopy(self.default_config))

        if 'experiment_config' in yaml_conf:
            self.exp_config = yaml_conf['experiment_config']

    def get_prebuilt_config(self):
        return self.config

    def create_player(self):
        return self.player_factory.create(self.algo_name, config=self.config, env=self.env)

    def create_agent(self, obs_space, action_space):
        return self.algo_factory.create(self.algo_name, base_name='run', observation_space=obs_space,
                                        action_space=action_space, config=self.config)

    def run(self, args):
        print('Started to play')
        player = self.create_player()
        player.restore(self.load_path)
        player.run()





