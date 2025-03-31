import json
from threading import Thread
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

    def train(self, n_epochs: int, data: 'RLData', batch_size = 64, gamma: float = 0.9, retarget: int = 10):
        self.__model.train()

        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        target_model = deepcopy(self.__model)

        for i in range(n_epochs):
            print(f"Epoch {i + 1}/{n_epochs}")

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

    def __init__(self, observations: tuple[tuple[int, ...], ...], actions: tuple[int, ...], rewards: tuple[int, ...], new_observations: tuple[tuple[int, ...]], ended: tuple[int]):
        self.__observations = observations
        self.__actions = actions
        self.__rewards = rewards
        self.__new_observations = new_observations
        self.__ended = ended

    def __len__(self):
        return len(self.__observations)

    def __getitem__(self, i) -> tuple[tuple[int, ...], int, float, tuple[int, ...], int]:
        return self.__observations[i], self.__actions[i], self.__rewards[i], self.__new_observations[i], self.__ended[i]

    @staticmethod
    def sample_data(env: gym.Env, n_episodes: int, policy: Optional[Policy] = None, seed=None) -> 'RLData':
        policy = policy if policy else Policy(env.observation_space, env.action_space, seed=seed)
        observations, actions, rewards, new_observations, ended = (), (), (), (), ()

        for i in range(n_episodes):
            obs, acts, rs, s_primes, terminated = PolicyAgent.sample_episode(env, policy, seed=seed)
            observations += obs
            actions += acts
            rewards += rs
            new_observations += s_primes
            ended += terminated

            if i % 1000 == 0:
                print(f"Sampled {i} episodes")
        print(len(observations))
        return RLData(observations, actions, rewards, new_observations, ended)

    def save(self, file_name: str):
        data_dict = {
            "observations": json.dumps(self.__observations),
            "actions": json.dumps(self.__actions),
            "rewards": json.dumps(self.__rewards),
            "new_observations": json.dumps(self.__new_observations),
            "ended": json.dumps(self.__ended)
        }
        with open(file_name, "w") as f:
            f.write(json.dumps(data_dict))

    @staticmethod
    def load(file_name: str) -> 'RLData':
        with open(file_name, "r") as f:
            data_dict = json.loads(f.read())

        # Make sure everything is tuple
        data_dict["observations"] = tuple(map(tuple, data_dict["observations"]))
        data_dict["actions"] = tuple(data_dict["actions"])
        data_dict["rewards"] = tuple(data_dict["rewards"])
        data_dict["new_observations"] = tuple(map(tuple, data_dict["new_observations"]))
        data_dict["ended"] = tuple(data_dict["ended"])

        return RLData(**data_dict)
