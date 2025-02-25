from functools import cache
from typing import Optional, Dict, Tuple
import gymnasium as gym
import numpy as np
from gymnasium.spaces import flatten_space, flatten
from gymnasium.core import ObsType, ActType
from board import Board
from constants import *
from functools import reduce
import json


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

        self._act_space: gym.spaces.Discrete = environment.action_space

        self._all_actions = range(self._act_space.n)

        self._transition_function = self.environment.env.env.P
        self._reward_function = self.environment.env.env.R

        if seed:
            self._act_space.seed(seed)
            self._obs_space.seed(seed)

        self._policy = policy if policy else self.__initialise_randomly()

    def __policy_evaluation(
        self, discount=0.1, stopping_criterion: float = 0.001
    ) -> Dict[Tuple[int, int, int, int], float]:
        """
        Uses an iterative algorithm to evaluate the value (v) of each state
        @param discount: Determines how much the agent values future rewards over the immediate rewards
        @param stopping_criterion: the threshold that determines when the algorithm stops
        @return: a dictionary mapping each observation in the environment to an associated value
        """
        # create the policy if it's not created yet
        policy = self._policy if self._policy else self.__initialise_randomly()

        # create a dictionary where the keys are all possible states of the environment
        all_v = {key: 0 for key in self.get_keys(self._obs_space)}

        # relates to the accuracy of our V. Is compared to stopping_criterion
        delta = np.inf

        # refrain all the values in the dictionary, until the difference between iterations is larger
        # than the threshold for the accuracy
        while delta >= stopping_criterion:
            delta = 0
            # the dictionary for the updated prediction for all state values
            new_all_v = all_v.copy()

            for state, action in policy.items():
                # if we are in a terminal state, it's value is always 0
                if not self._transition_function[state, action]:
                    new_all_v[state] = 0
                    continue

                v = all_v[state]
                state_chance = 1 / len(self._transition_function[state, action])

                new_v = sum(
                    tuple(
                        state_chance * (self._reward_function[new_state] + discount * all_v[new_state])
                        for new_state in self._transition_function[state, action]
                    )
                )

                delta = max(delta, abs(new_v - v))
                new_all_v[state] = new_v
            all_v = new_all_v

        return all_v

    def value_iteration(self, discount = 0.1, stopping_criterion = 0.001):
        # create the policy if it's not created yet
        policy = self._policy if self._policy else self.__initialise_randomly()

        # create a dictionary where the keys are all possible states of the environment
        all_v = {key: 0 for key in self.get_keys(self._obs_space)}

        # relates to the accuracy of our V. Is compared to stopping_criterion
        delta = np.inf

        while delta >= stopping_criterion:
            delta = 0

            new_all_v = all_v.copy()

            for state, action in policy.items():

                best_val = float('-inf')

                if not self._transition_function.get((state, action)):
                    new_all_v[state] = 0
                    continue

                v = all_v[state]
                state_chance = 1 / len(self._transition_function[state, action])

                new_v = sum(
                    tuple(
                        state_chance * (self._reward_function[new_state] + discount * all_v[new_state])
                        for new_state in self._transition_function[state, action]
                    )
                )

                best_val = max(best_val, new_v)

                delta = max(delta, abs(new_v - v))
                new_all_v[state] = best_val

            all_v = new_all_v

        return Policy(self.environment, policy = policy)

    def __improve(self, discount: float, stopping_criterion: float) -> tuple['Policy', bool]:
        """
        Performs one step of policy improvement.

        @param discount: The discount used when solving the Bellman equations.
        @param stopping_criterion: Stopping criterion used for stopping when evaluating policy.
        @return: A tuple of the new policy, and if it changed with regard to the current policy.
        """
        # Initialise new policy, so old policy is not modified. Nice to keep current policy immutable.
        policy = Policy(self.environment)
        made_changes = False
        transition_function = self.environment.env.env.P

        value_dict = self.__policy_evaluation(discount=discount, stopping_criterion=stopping_criterion)

        # Try to improve all states
        for state in self:
            values = {}

            # Compute the expected value for all actions
            for action in self._all_actions:
                action_value = sum(
                    map(
                        lambda new_state: self._reward_function[new_state] + value_dict[new_state] * discount,  # Get value for each successor
                        transition_function[(state, action)]  # Get all possible successors for action
                    )
                )  / len(transition_function[(state, action)])  # Compute average value for action

                values[action] = action_value

            # Argmax
            new_action = max(values.items(), key=lambda kv: kv[1])[0]

            # Check if changes were made and change policy
            made_changes = new_action != self[state] or made_changes
            policy[state] = new_action

        return policy, made_changes

    def policy_iterate(self, discount: float = 0.1, evaluation_stopping_criterion: float = 0.001) -> 'Policy':
        """
        Performs policy iteration.

        @param discount: The discount used when solving the Bellman equations.
        @param evaluation_stopping_criterion: The stopping criterion used when evaluating the policy.
        @param max_iter: The maximum number of iterations before it stops. -1 means go until policy converges.
        @return: The optimised policy.
        """
        made_changes, policy = True, self

        # Loop while policy has not converged
        while made_changes:
            policy, made_changes = policy.__improve(discount=discount, stopping_criterion=evaluation_stopping_criterion)
        return policy

    def __initialise_randomly(self) -> dict[tuple, ActType]:
        """
        Initialises a random policy.

        @return: A random policy dictionary
        """
        # Assign a random action from the action space to each key, as long as the key does not belong to a terminal state.
        return {key: self._act_space.sample() for key in self.get_keys(self._obs_space) if len(self._transition_function[key, 0]) > 0}

    def __getitem__(self, observation: ObsType | Tuple[int, ...]) -> ActType:
        """
        Takes an observation and maps it to the corresponding action.

        @param observation: Observation of the environment.
        @return: Action to take under this policy.
        """
        key = tuple(flatten(self._obs_space, observation)) if not isinstance(observation, tuple) else observation
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

    def save(self, file_name: str) -> None:
        """
        Saves a policy under a certain name.

        @param file_name: The file name under which the policy is saved.
        """
        with open(file_name, "w") as f:
            json_dict = {str(key): value for key, value in self.items()}
            f.write(json.dumps(json_dict))

    @staticmethod
    def load(environment: gym.Env, file_name: str) -> 'Policy':
        """
        Tries to load a policy from a json file.

        @param environment: The environment the policy describes.
        @param file_name: The file where the policy can be found.

        @return: The constructed policy
        """
        with open(file_name, "r") as f:
            txt = json.loads(f.read())
        policy_dict = {tuple(map(int, filter(lambda x: x.isnumeric(), tuple(key)))): value for key, value in txt.items()}
        return Policy(environment, policy=policy_dict)
