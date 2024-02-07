# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PettingZoo interface to meltingpot environments."""

import functools
import numpy as np
from gymnasium import utils as gym_utils
import matplotlib.pyplot as plt
from ml_collections import config_dict
from pettingzoo import utils as pettingzoo_utils
from pettingzoo.utils import wrappers

from examples import utils
from meltingpot.python import substrate

PLAYER_STR_FORMAT = 'player_{index}'
MAX_CYCLES = 1000


def parallel_env(env_config, max_cycles=MAX_CYCLES):
  return _ParallelEnv(env_config, max_cycles)


def raw_env(env_config, max_cycles=MAX_CYCLES):
  return pettingzoo_utils.parallel_to_aec_wrapper(
      parallel_env(env_config, max_cycles))


def env(env_config, max_cycles=MAX_CYCLES):
  aec_env = raw_env(env_config, max_cycles)
  aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
  aec_env = wrappers.OrderEnforcingWrapper(aec_env)
  return aec_env


class _MeltingPotPettingZooEnv(pettingzoo_utils.ParallelEnv):
  """An adapter between Melting Pot substrates and PettingZoo's ParallelEnv."""

  def __init__(self, env_config, max_cycles):
    """
    :param env_config: A config dict for the environment.
    :param max_cycles: The maximum number of cycles to run the environment for.
    :param reward_type: The type of reward to use. Either "shared" or "individual".
    """
    self.env_config = config_dict.ConfigDict(env_config)
    self.max_cycles = max_cycles
    self.reward_type = env_config.reward_type
    self._env = substrate.build_from_config(self.env_config, roles =self.env_config.default_player_roles)
    self._num_players = len(self._env.observation_spec())
    self.possible_agents = [
        PLAYER_STR_FORMAT.format(index=index)
        for index in range(self._num_players)
    ]
    self.individual_observation_names = env_config.individual_observation_names
    observation_space = utils.remove_world_observations_from_space(
        utils.spec_to_space(self._env.observation_spec()[0]))
    # lru_cache is used to cache the observation and action spaces
    # if the agent_id is the same.
    self.observation_space = functools.lru_cache(
        maxsize=None)(lambda agent_id: observation_space)
    action_space = utils.spec_to_space(self._env.action_spec()[0])
    self.action_space = functools.lru_cache(maxsize=None)(
        lambda agent_id: action_space)
    self.state_space = utils.spec_to_space(
        self._env.observation_spec()[0]['WORLD.RGB'])

  def state(self):
    return self._env.observation()

  def reset(self, seed=None):
    """See base class."""
    timestep = self._env.reset()
    self.agents = self.possible_agents[:]
    self.num_cycles = 0
    self.step_count = 0
    return utils.timestep_to_observations(timestep, self.individual_observation_names)

  def step(self, action):
    """See base class."""
    if isinstance(action, dict):
      actions = [action[agent] for agent in self.agents]
    else:
      actions = [action[agent] for agent in range(self.num_agents)]
    timestep = self._env.step(actions)
    rewards = {
        agent: timestep.reward[index] for index, agent in enumerate(self.agents)
    }
    self.num_cycles += 1
    self.step_count += 1
    done = timestep.last() or self.num_cycles >= self.max_cycles
    dones = {agent: done for agent in self.agents}
    infos = {agent: {} for agent in self.agents}
    if done:
      self.agents = []

    if self.reward_type == "shared":
      rewards = sum([rewards[reward_key] for reward_key in rewards.keys()])/len(rewards) 
    observations = utils.timestep_to_observations(timestep, self.individual_observation_names)
    return observations, rewards, dones, infos

  def close(self):
    """See base class."""
    self._env.close()

  def render(self, mode='human', filename=None):
    rgb_arr = self.state()[0]['WORLD.RGB']
    if mode == 'human':
      plt.cla()
      plt.imshow(rgb_arr, interpolation='nearest')
      if filename is None:
        plt.show(block=False)
      else:
        plt.savefig(filename)
      return None
    return rgb_arr

  
  def get_obs(self):
    return [np.moveaxis(self._env.observation()[agent_id]["RGB"], -1, 0) for agent_id in range(self.num_agents)]
  def get_state(self):
    obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
              np.float32
          )
    return obs_concat
  def get_avail_actions(self):
    test = [np.ones(self._env.action_spec()[agent_id].num_values, dtype = self._env.action_spec()[agent_id].dtype) for agent_id in range(self.num_agents)]
    return test
  def get_state_size(self):
    """Returns the size of the global state."""
    obs_shape = self._env.observation()[0]["RGB"].shape
    return (obs_shape[2] * self.num_agents, obs_shape[0], obs_shape[1])


class _ParallelEnv(_MeltingPotPettingZooEnv, gym_utils.EzPickle):
  metadata = {'render_modes': ['human', 'rgb_array']}

  def __init__(self, env_config, max_cycles):
    gym_utils.EzPickle.__init__(self, env_config, max_cycles)
    _MeltingPotPettingZooEnv.__init__(self, env_config, max_cycles)
