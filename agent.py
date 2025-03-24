from abc import abstractmethod
from copy import deepcopy
from functools import reduce
from itertools import pairwise
from operator import add
from typing import Any, SupportsFloat, Optional
import gymnasium as gym
from gymnasium.core import ObsType
import numpy as np
import torch
from torch import nn
from policy import Policy
from temporal_difference_policy import epsilon_greedy


class Agent:
    def __init__(self, env: gym.Env):
        self.__reward: SupportsFloat = 0
        self._action_space = env.action_space
        self._obs_space = env.observation_space

        self.__env: gym.Env = env
        self._obs: Optional[ObsType] = None
        self._terminated = False

    def __generate_move(
        self,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        move = self._generate_move_helper()
        print(move)
        observation, reward, terminated, truncated, info = self.__env.step(move)

        self.__reward += reward
        self._obs = observation
        self._terminated = terminated

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def run_agent(self, initial_obs: ObsType) -> SupportsFloat:
        self._obs = initial_obs
        while not self.terminated:
            self.__generate_move()
        return self.reward

    def get_env_str(self) -> str:
        print("A")
        return str(self.__env)

    # Abstract methods
    @abstractmethod
    def _generate_move_helper(self) -> int:
        """
        Generates a move given the current observation.

        @returns: An integer representing the move taken.
        """
        pass

    # Properties
    @property
    def reward(self) -> SupportsFloat:
        return self.__reward

    @property
    def terminated(self) -> bool:
        return self._terminated


class RandomAgent(Agent):

    def __init__(self, env: gym.Env, seed=None):
        super().__init__(env)
        if seed:
            np.random.seed(seed)

    def _generate_move_helper(self) -> int:
        return self._action_space.sample()


class PolicyAgent(Agent):

    def __init__(self, env: gym.Env, policy: Optional[Policy | None] = None):
        super().__init__(env)
        self.__policy = policy if policy else Policy(env)

    def _generate_move_helper(self) -> int:
        return self.__policy[self._obs]


class HumanAgent(Agent):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        print(self.get_env_str())

    def _generate_move_helper(self) -> int:
        direction = input("Where would you like to go (left/right/up/down)?\n:\t")
        match direction:
            case "right":
                return 2
            case "left":
                return 0
            case "up":
                return 1
            case "down":
                return 3
            case _:
                print("Invalid direction. Please try again:")
                return self._generate_move_helper()


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

    def train(self, env: gym.Env, n_episodes: int, seed = None, gamma: float = 0.9, retarget: int = 10, epsilon = 0.5):
        self.__model.train()

        for i in range(n_episodes):
            print(i)
            obs, _ = env.reset(seed=seed)
            obs = self._obs_to_tensor(obs)

            target_model = deepcopy(self.__model)
            terminated, truncated = False, False
            t = 0
            while not (terminated or truncated):
                self.__optimiser.zero_grad()
                qs = self.__model.forward(obs)
                action = epsilon_greedy({act: qs[act] for act in range(self._action_space.n)}, epsilon=epsilon)

                new_obs, reward, terminated, truncated, _ = env.step(action)
                new_obs = self._obs_to_tensor(new_obs)
                target_qs = target_model.forward(new_obs)

                target = reward + gamma * torch.max(target_qs)
                loss = (target - qs[action]) ** 2
                loss.backward()
                self.__optimiser.step()

                t += 1
                if t % retarget == 0:
                    target_model = deepcopy(self.__model)

                obs = new_obs

        self.__model.eval()

    def _generate_move_helper(self) -> int:
        with torch.no_grad():
            obs = self._obs_to_tensor(self._obs)
            return int(torch.argmax(self.__model.forward(obs)))

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
