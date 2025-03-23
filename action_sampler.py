import json
from copy import deepcopy
from typing import Optional
from functools import reduce
from gymnasium.core import ActType
import numpy as np


class ActionSampler:
    """
    An ActionSampler is the value of a Policy. It determines at what probability each action in the policy will be
    sampled. This ActionSampler can sample actions with any predetermined probabilities.
    """

    def __init__(self, action_probabilities: dict[int, float], generator: Optional[np.random._generator] = None):
        """
        Constructs the CustomProbabilityActionSampler.

        @param action_probabilities: A dictionary mapping all actions to their respective probabilities. Probabilities are
                                     normalised automatically.
        @param generator: Random generator used to sample actions.
        """
        # Normalise probabilities.
        self.__probabilities = dict(map(lambda kv: (kv[0], kv[1] / sum(action_probabilities.values())),
                                        action_probabilities.items()))
        self.__generator = generator if generator else np.random.default_rng()

    def sample(self) -> int:
        """
        Samples an action.

        @return: The sampled action.
        """
        n = self.__generator.uniform(0, 1)
        return reduce(lambda x, y: x if x[1] >= n else (y[0], y[1] + x[1]), self.__probabilities.items())[0]

    def greedy_sample(self) -> int:
        """
        Greedily samples an action, returning the action with the highest probability.

        @return: The action with the highest probability.
        """
        return max(self.__probabilities.items(), key=lambda item: item[1])[0]

    @property
    def probabilities(self) -> dict[int, float]:
        """
        @return: A dictionary of each action mapping to the probability of that action.
        """
        return dict(deepcopy(self.__probabilities))

    @staticmethod
    def load(txt: str) -> 'ActionSampler':
        """
        Tries to load an ActionSampler from a string, should be a json format mapping an action to a probability.

        @param txt: The string in json format mapping action to probability.
        @return: A CustomProbabilityActionSampler with the predefined probabilities.
        """
        probabilities = {int(action): float(probability) for action, probability in json.loads(txt).items()}
        return ActionSampler(probabilities)


class UniformActionSampler(ActionSampler):
    """
    This ActionSampler samples all action with uniform probabilities.
    """

    def __init__(self, act_space: tuple[int, ...], generator: Optional[np.random._generator] = None):
        """
        Constructs an ActionSampler.

        @param act_space: A tuple of all actions this action sampler can return.
        @param generator: The numpy random generator used to sample actions.
        """

        probabilities = {action: 1 / len(act_space) for action in act_space}
        super().__init__(probabilities, generator)


class DeterministicAction(ActionSampler):
    """
    This ActionSampler is deterministic, so it always returns the action specified. Exists for compatibility.
    """

    def __init__(self, action: int):
        """
        Constructs the DeterministicActionSampler.

        @param action: The action that this Sampler will always return.
        """
        super().__init__({action: 1})


class EpsilonGreedyActionSampler(ActionSampler):
    """
    This ActionSampler samples a given action at a probability of epsilon, while in all other cases taking a random
    sample from the action space.
    """

    def __init__(self, act_space: tuple[int], optimal_action: int, epsilon: float, generator: Optional[np.random._generator] = None):
        """
        Constructs and EpsilonSoftActionSampler.

        @param act_space: The action space from which the actions will be sampled.
        @param optimal_action: The optimal action, picked with probability epsilon.
        @param epsilon: The probability the optimal action will be picked.
        @param generator: Random generator used to sample actions.
        """
        assert 0 <= epsilon <= 1, "Epsilon not in range [0, 1]"
        assert optimal_action in act_space, "Optimal action is not in action space"
        probabilities = {action: epsilon / len(act_space) for action in act_space}
        probabilities[optimal_action] = 1 - epsilon + epsilon / len(act_space)

        super().__init__(probabilities, generator)
