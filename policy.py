from functools import cache
from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten_space, flatten
from gymnasium.core import ObsType, ActType


class Policy:
    """
    A policy is a fancy dictionary that maps a flattened observation to an action.
    """

    def __init__(self, environment: gym.Env, policy: Optional[dict] = None, seed: Optional[int | None] = None):
        """
        Constructs the policy.

        @param environment: The environment this policy provides a policy for.
        @param policy: Optional, a dictionary mapping each observation in the environment to an action in the action_space
                       If not present, creates a random policy.
        """
        self._obs_space = environment.observation_space

        if environment.observation_space.is_np_flattenable:
            self._obs_space_flat = flatten_space(environment.observation_space)
        else:
            raise NotImplementedError

        self._act_space = environment.action_space

        if seed:
            self._act_space.seed(seed)
            self._obs_space.seed(seed)
            self._obs_space_flat.seed(seed)

        self._policy = policy if policy else self.__initialise_randomly()

    def __initialise_randomly(self) -> dict[tuple, ActType]:
        """
        Initialises a random policy.
        """
        policy = {}
        for key in self.get_keys():
            policy[key] = self._act_space.sample()
        return policy

    def __getitem__(self, observation: ObsType) -> ActType:
        """
        Takes an observation and maps it to the corresponding action.

        @param observation: Observation of the environment.
        @return: Action to take under this policy.
        """
        key = tuple(flatten(self._obs_space, observation))
        return self._policy[key]

    def keys(self):
        """
        @return: All possible observations of the environment as flattened tuples.
        """

        return self._policy.keys()

    def values(self):
        """
        @return: The actions the Policy maps to.
        """
        return self._policy.values()

    def items(self):
        """
        @return: All key value pairs.
        """
        return self._policy.items()

    @staticmethod
    @cache
    def __get_keys(bounds: tuple[tuple[int, ...]]) -> tuple[tuple[int,...]]:
        """
        Gets all possible combinations of observations as a tuple of tuples.

        @param bounds: The bounds of the observation space as a nested tuple in the form ((lower1, upper1), (lower2, upper2), ...)
        @return: All possible observations as a tuple of tuples, each tuple having length bounds.shape[0].
        """
        if len(bounds) == 0:
            return ((),)

        previous = Policy.__get_keys(bounds[1:])
        final = ()

        for i in range(bounds[0][0], bounds[0][1] + 1):
            final += tuple(map(lambda tup: (i,) + tup, previous))

        return final

    def get_keys(self) -> tuple[tuple[int,...]]:
        """
        Gets all possible combinations of observations as a tuple of tuples.

        @return: All possible observations as a tuple of tuples, each tuple having length bounds.shape[0].
        """
        bounds = tuple(map(tuple, np.stack((self._obs_space_flat.low, self._obs_space_flat.high)).T))
        return Policy.__get_keys(bounds)
