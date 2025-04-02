from abc import abstractmethod
from typing import Any, SupportsFloat, Optional
import gymnasium as gym
from gymnasium.core import ObsType
import numpy as np
from numpy.f2py.rules import options

from policy import Policy


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
        print(self.__env.render())
        self._obs = initial_obs
        while not self.terminated:
            self.__generate_move()
        return self.reward

    def get_env_str(self) -> str:
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
        self.__policy = (
            policy if policy else Policy(env.observation_space, env.action_space)
        )

    def _generate_move_helper(self) -> int:
        return self.__policy.get_action(self._obs)

    @staticmethod
    def sample_episode(
        env: gym.Env, policy: Policy, seed=None, reset_options: dict[str, Any] = None
    ) -> tuple[
        tuple[tuple[int, ...]],
        tuple[int, ...],
        tuple[float, ...],
        tuple[tuple[int, ...]],
        tuple[int, ...],
    ]:
        """
        Samples an episode from start to finish from the given environment under the given policy.

        @param env: Environment the episodes are sampled from.
        @param policy: Policy used for action generation.
        @param seed: Random seed used, optional.
        @param reset_options: Options used for resetting the environment.
        @return: A tuple of the form (States, Actions, Rewards, New States, Terminateds)
        """

        # Reset environment and initialise variables.
        obs, _ = env.reset(seed=seed, options=reset_options)
        obs = tuple(map(int, gym.spaces.flatten(env.observation_space, obs)))
        terminated, truncated = False, False
        observations, actions, rewards, new_observations, ended = (), (), (), (), ()

        # Loop until episode finishes.
        while not (terminated or truncated):
            # Generate action under given policy.
            actions += (policy.get_action(obs),)

            # Take action.
            new_obs, reward, terminated, truncated, _ = env.step(actions[-1])

            # Convert observation to proper form.
            new_obs = tuple(
                map(int, gym.spaces.flatten(env.observation_space, new_obs))
            )

            # Save all needed data.
            observations += (obs,)
            rewards += (reward,)
            new_observations += (new_obs,)
            obs = new_obs
            ended += (int(terminated or truncated),)

        return observations, actions, rewards, new_observations, ended


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
