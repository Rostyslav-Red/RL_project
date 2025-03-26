from typing import Optional
from torch.utils.data import DataLoader
from agent import Agent, PolicyAgent
from copy import deepcopy
from functools import reduce
from itertools import pairwise
from operator import add
import gymnasium as gym
from gymnasium.core import ObsType
import torch
from torch import nn
from policy import Policy
import numpy as np

class DeepQLearningAgent(Agent):

    def __init__(self, env: gym.Env, model: nn.Module, lr=0.01):
        super().__init__(env)
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = model.to(self.__device)
        self.__criterion = nn.MSELoss()
        self.__optimiser = torch.optim.SGD(self.__model.parameters(), lr=lr)

    @property
    def model(self):
        return self.__model

    def train(self, env: gym.Env, n_epochs: int, n_episodes = 10, batch_size = 64, policy: Optional[Policy] = None,
              seed = None, gamma: float = 0.9, retarget: int = 10):
        self.__model.train()

        policy = policy if policy else Policy(env.observation_space, env.action_space)
        data = RLData(env, n_episodes, policy, seed=seed)
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        target_model = deepcopy(self.__model)

        for i in range(n_epochs):
            t = 0

            for obs, action, reward, new_obs, ended in loader:
                self.__optimiser.zero_grad()
                obs = torch.tensor(np.array(obs)).to(self.__device).float().T
                new_obs = torch.tensor(np.array(new_obs)).to(self.__device).float().T
                reward = reward.to(self.__device)
                action = action.to(self.__device)
                ended = ended.to(self.__device)

                qs = self.__model.forward(obs)
                target_qs = target_model.forward(new_obs)

                target = (reward + gamma * torch.max(target_qs, dim=1)[0] * (1 - ended)).detach()
                loss = ((target - qs.gather(1, action.unsqueeze(1)).squeeze(1)) ** 2).mean()
                loss.backward()
                self.__optimiser.step()

                t += 1
                if t % retarget == 0:
                    target_model.load_state_dict(self.__model.state_dict())

            print(i)

        self.__model.eval()

    @staticmethod
    def generate_buffer(env: gym.Env, n_episodes: int, policy: Policy, seed = None) -> tuple:
        return tuple(PolicyAgent.sample_episode(env, policy, seed=seed) for _ in range(n_episodes))

    def _generate_move_helper(self) -> int:
        with torch.no_grad():
            obs = self._obs_to_tensor(self._obs)
            return int(self.__model.forward(obs).argmax())

    def _obs_to_tensor(self, obs: ObsType) -> torch.Tensor:
        return torch.Tensor(tuple(map(int, gym.spaces.flatten(self._obs_space, obs)))).to(self.__device)

    def save(self, file_name: str):
        torch.save(self.__model, file_name)

    @staticmethod
    def build_model(env: gym.Env, layer_sizes: tuple[int, ...], lr=0.01) -> 'DeepQLearningAgent':
        input_size = len(gym.spaces.flatten_space(env.observation_space).low)
        output_size = int(env.action_space.n)

        layer_sizes = (input_size,) + layer_sizes + (output_size,)
        layers = tuple(reduce(add, map(lambda x: (nn.Linear(x[0], x[1]), nn.ReLU()), pairwise(layer_sizes))))

        model = nn.Sequential(*layers[:-1])
        return DeepQLearningAgent(env, model, lr=lr)

    @staticmethod
    def load(environment: gym.Env, file_name: str) -> 'DeepQLearningAgent':
        model = torch.load(file_name, weights_only=False)
        model.eval()
        return DeepQLearningAgent(environment, model)


class RLData(torch.utils.data.Dataset):

    def __init__(self, env: gym.Env, n_episodes: int, policy: Policy, seed=None):
        self.__observations = ()
        self.__actions = ()
        self.__rewards = ()
        self.__new_observations = ()
        self.__ended = ()
        for _ in range(n_episodes):
            observations, actions, rewards, new_observations, ended = PolicyAgent.sample_episode(env, policy, seed=seed)
            self.__observations += observations
            self.__actions += actions
            self.__rewards += rewards
            self.__new_observations += new_observations
            self.__ended += ended

    def __len__(self):
        return len(self.__observations)

    def __getitem__(self, i) -> tuple[tuple[int, ...], int, float, tuple[int, ...], bool]:
        return self.__observations[i], self.__actions[i], self.__rewards[i], self.__new_observations[i], self.__ended[i]
