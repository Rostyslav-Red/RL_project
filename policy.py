from functools import cache
from typing import Optional, Dict, Tuple
import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten_space, flatten
from gymnasium.core import ObsType, ActType
from board import Board
from constants import *


class Policy:
    """
    A policy is a fancy dictionary that maps a flattened observation to an action.
    """

    def __init__(
        self,
        environment: gym.Env,
        policy: Optional[dict] = None,
        seed: Optional[int | None] = None,
    ):
        """
        Constructs the policy.

        @param environment: The environment this policy provides a policy for.
        @param policy: Optional, a dictionary mapping each observation in the environment to an action in the action_space
                       If not present, creates a random policy.
        """
        self.environment = environment

        self._obs_space = environment.observation_space

        self._act_space = environment.action_space

        if seed:
            self._act_space.seed(seed)
            self._obs_space.seed(seed)

        self._policy = policy if policy else self.__initialise_randomly()

    def __policy_evaluation(
        self, discount=0.1, stopping_criterion: float = 0.001
    ) -> Dict[Tuple[int, int, int, int], float]:
        """
        Uses an iterative algorithm to evaluate the value (v) of each state
        :param discount: Determines how much the agent values future rewards over the immediate rewards
        :param stopping_criterion: the threshold that determines when the algorithm stops
        :return: a dictionary mapping each observation in the environment to an associated value
        """

        # Describes the dynamics of the environment. Watch Board.P
        P = self.environment.env.env.P
        # Described the reward space of hte environment. Watch Board.R
        R = self.environment.env.env.R

        # create the policy if it's not created yet
        policy = self._policy if self._policy else self.__initialise_randomly()

        # create a dictionary where the keys are all possible states of the environment
        all_v = {key: 0 for key in policy.keys()}

        # relates to the accuracy of our V. Is compared to stopping_criterion
        delta = np.inf

        # refrain all the values in the dictionary, until the difference between iterations is larger
        # than the threshold for the accuracy
        while delta >= stopping_criterion:
            delta = 0
            # the dictionary for the updated prediction for all state values
            new_all_v = {}

            for state, action in policy.items():
                # if we are in a terminal state, it's value is always 0
                if not P[(state, action)]:
                    new_all_v[state] = 0
                    continue

                v = all_v[state]
                state_chance = 1 / len(P[state, action])

                new_v = sum(
                    tuple(
                        state_chance * (R[new_state] + discount * all_v[new_state])
                        for new_state in P[(state, action)]
                    )
                )

                delta = max(delta, abs(new_v - v))
                new_all_v[state] = new_v
            all_v = new_all_v
        return all_v

    def __initialise_randomly(self) -> dict[tuple, ActType]:
        """
        Initialises a random policy.

        @return: A random policy dictionary
        """
        policy = {}
        for key in self.get_keys(self._obs_space):
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

    def __setitem__(self, key: ObsType, value: ActType) -> None:
        """
        Sets the action an observation maps to, to a different value.

        @param key: The observation for which the action is changed.
        @param value: The new action.
        """
        self._policy[key] = value

    def __iter__(self):
        """
        @return: In iterator over the keys of the policy.
        """
        return self._policy.__iter__()

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
    def __get_keys(bounds: tuple[tuple[int, ...]]) -> tuple[tuple[int, ...]]:
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

    @staticmethod
    def get_keys(obs_space: gym.Space) -> tuple[tuple[int, ...]]:
        """
        Gets all possible combinations of observations as a tuple of tuples.

        @param: obs_space: The observation space of the environment.
        @return: All possible observations as a tuple of tuples, each tuple having length bounds.shape[0].
        """
        if not obs_space.is_np_flattenable:
            raise NotImplementedError(
                "Policy keys for non-flattenable observation are not defined."
            )

        flat_obs_space = flatten_space(obs_space)
        bounds = tuple(
            map(tuple, np.stack((flat_obs_space.low, flat_obs_space.high)).T)
        )
        return Policy.__get_keys(bounds)
