from abc import abstractmethod
from typing import Any, SupportsFloat, Optional
import gymnasium as gym
from gymnasium.core import ObsType
import numpy as np


class Agent:
    def __init__(self, env: gym.Env):
        self.__reward: SupportsFloat = 0
        self._action_space = env.action_space

        self.__env: gym.Env = env
        self._obs: Optional[ObsType] = None
        self._terminated = False

    def __generate_move(self) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        move = self._generate_move_helper()
        observation, reward, terminated, truncated, info = self.__env.step(move)

        self.__reward += reward
        self._obs = observation
        self._terminated = False

        return observation, reward, terminated, truncated, info,

    def run_agent(self, initial_obs: ObsType) -> SupportsFloat:
        self._obs = initial_obs
        while not self.terminated:
            self.__generate_move()
            print(self.__env)
        return self.reward


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

