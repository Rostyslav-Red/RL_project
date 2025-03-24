import warnings
from functools import cache, reduce
from itertools import groupby
from typing import Optional, Tuple, Iterable
import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten
from gymnasium.core import ObsType, ActType
from action_sampler import ActionSampler
from constants import *
import json


class Policy:
    """
    A policy is a fancy dictionary that maps a flattened observation to an action.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        policy: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        """
        Constructs the policy.

        @param obs_space: The observation space of the environment.
        @param act_space: The action space of the environment.
        @param policy: Optional, a dictionary mapping each observation in the environment to an action in the action_space
                       If not present, creates a random policy.
        @param seed: The random seed used to sample actions.
        """
        self._obs_space = obs_space

        self._act_space: gym.spaces.Discrete = act_space

        self._all_actions = tuple(range(self._act_space.n))

        self._rng = np.random.default_rng(seed)

        if seed:
            self._act_space.seed(seed)
            self._obs_space.seed(seed)

        # Check policy validity
        policy_valid = self.is_valid(policy, self._obs_space, self._act_space)
        if policy and not policy_valid:
            warnings.warn("Invalid policy passed, randomising policy instead.")

        self._policy = policy if policy_valid else self.__initialise_randomly()

    # Dunder methods
    def __getitem__(self, observation: ObsType | Tuple[int, ...]) -> ActionSampler:
        """
        Takes an observation and maps it to the corresponding action.

        @param observation: Observation of the environment.
        @return: Action to take under this policy.
        """
        key = self._obs_to_tuple(observation) if not isinstance(observation, tuple) else observation
        return self._policy[key]

    def __setitem__(self, key: ObsType, value: int | ActionSampler) -> None:
        """
        Sets the action an observation maps to, to a different value.

        @param key: The observation for which the action is changed.
        @param value: The new action.
        """
        if isinstance(value, int):
            self._policy[key] = ActionSampler.deterministic_action_sampler(value)
        elif isinstance(value, ActionSampler):
            self._policy[key] = value
        else:
            raise ValueError(f"Policy value needs to be either an ActionSampler or integer, but is of type {type(value)}")

    def __iter__(self):
        """
        @return: In iterator over the keys of the policy.
        """
        return self._policy.__iter__()

    # Dict methods
    def keys(self) -> Iterable[tuple[int, ...]]:
        """
        @return: All possible observations of the environment as flattened tuples.
        """
        return self._policy.keys()

    def values(self) -> Iterable[ActionSampler]:
        """
        @return: The actions the Policy maps to.
        """
        return self._policy.values()

    def items(self) -> Iterable[tuple[tuple[int,...], ActionSampler]]:
        """
        @return: All key value pairs.
        """
        return self._policy.items()

    def get_action(self, state: ObsType | Tuple[int, ...]) -> ActType:
        return self[state].sample()

    # Misc methods
    def __initialise_randomly(self) -> dict[tuple, ActType]:
        """
        Initialises a random policy.

        @return: A random policy dictionary
        """
        # Assign a random action from the action space to each key, as long as the key does not belong to a terminal state.
        return {key: ActionSampler.uniform_action_sampler(self._all_actions, generator=self._rng) for key in self.get_keys(self._obs_space)}

    @staticmethod
    def _action_to_value(q: dict[tuple[tuple[int, ...], int], float], observation: tuple[int, ...]) -> dict[int, float]:
        """
        Given a dictionary mapping observation + action to a value, and an observation, gives back a dictionary of
        action mapping to value.

        @param q: Dictionary mapping observation + action to value.
        @param observation: Observation for which the dict needs to be computed.
        @return: A dictionary mapping all actions in the observation to a value.
        """
        return {key[1]: value for key, value in q.items() if observation == key[0]}

    def _greedy_policy_from_q(self, q: dict[tuple[tuple[int, ...], int], float]) -> 'Policy':
        # Create a tuple of the form (obs, (((obs, act), val), ...)), grouping same observations together
        obs_to_obs_act_val = groupby(q.items(), lambda kkv: kkv[0][0])
        policy = dict(
            map(lambda x: (x[0],  # x[0] is the observation
                           ActionSampler.deterministic_action_sampler(max(x[1],  # x[1] is the tuple of ((obs, act), val), ...), taking max over it gives tuple with highest val.
                                                            key=lambda y: y[1])[0][1]  # y[1] is the value, [0][1] on max gives back the action
                                                        )),
                obs_to_obs_act_val))
        return Policy(self._obs_space, self._act_space, policy=policy)

    def _obs_to_tuple(self, observation: ObsType) -> tuple[int, ...]:
        return tuple(map(int, flatten(self._obs_space, observation)))

    # Static methods
    @staticmethod
    def is_valid(policy_dict: dict[tuple[int, ...], ActionSampler], obs_space: gym.Space, act_space: gym.Space) -> bool:
        """
        Checks if the policy-like dictionary is a valid policy (all possible observations are in dict keys and all
        values are ActionSamplers with all actions in the action space.

        @param policy_dict: The policy-like dictionary.
        @param obs_space: The observation space of the environment this policy should describe.
        @param act_space: The action space of the environment this policy should describe.
        @return: A boolean, true if the policy is valid, false otherwise.
        """
        policy_valid = False
        if policy_dict:
            keys = set(Policy.get_keys(obs_space))
            # If union of two sets is same length as set before union, the policy-like dictionary accounts for all
            # observations in the obs_space
            keys_valid = len(keys.union(policy_dict.keys())) == len(keys)

            # Check if any value violates the requirements for the policy value: all values are ActionSamplers, all
            # ActionSampler keys are in the action_space
            vals_valid = len(tuple(
                filter(lambda act_sampler:  # Check for every act_sampler if: any key is not in act_space, is an ActionSampler
                       not isinstance(act_sampler, ActionSampler) or
                       # Check if any key in not in act_space
                       len(tuple(filter(lambda key: not act_space.contains(key), act_sampler.probabilities.keys()))) != 0,
                       policy_dict.values()))) == 0  # If filter returns 0, no violations of requirements is found.
            policy_valid = keys_valid and vals_valid
        return policy_valid

    @staticmethod
    @cache
    def __get_keys(bounds: tuple[tuple[int, ...]]) -> tuple[tuple[int, ...]]:
        """
        Gets all possible combinations of observations as a tuple of tuples.

        @param bounds: The bounds of the observation space as a nested tuple in the form ((lower1, upper1), (lower2, upper2), ...)
        @return: All possible observations as a tuple of tuples, each tuple having length bounds.shape[0].
        """
        # Base case, no bounds so no keys returned.
        if len(bounds) == 0:
            return ((),)

        # Recursive case, get bounds of the tail
        previous = Policy.__get_keys(bounds[1:])

        # For each number from upper to lower bound, prepend it to all previous tuples, and add these tuples together
        # into one big tuple.
        final = reduce(lambda x, i:
                       x + tuple(map(lambda tup: (i,) + tup, previous)),
                       range(bounds[0][0], bounds[0][1] + 1), ())

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

    # Policy loading and saving
    def save(self, file_name: str) -> None:
        """
        Saves a policy under a certain name.

        @param file_name: The file name under which the policy is saved.
        """
        with open(file_name, "w") as f:
            json_dict = {str(key): json.dumps(value.probabilities) for key, value in self.items()}
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
        policy_dict = {tuple(map(int, filter(lambda x: x.isnumeric(), tuple(key)))): ActionSampler.load(value) for key, value in txt.items()}
        return Policy(environment.observation_space, environment.action_space, policy=policy_dict)
