from policy import Policy
import gymnasium as gym
from typing import Dict, Tuple, Optional, cast, Union
import numpy as np
from board import Board
from constants import *
from copy import deepcopy


class DynamicProgrammingPolicy(Policy):
    """
    Produces policies through Policy Iteration or Value Iteration
    """

    class DynamicProgrammingAlgorithms(str, Enum):
        PolicyIteration = "PolicyIteration"
        ValueIteration = "ValueIteration"

    def __init__(
        self,
        environment: gym.Env,
        policy: Optional[dict] = None,
        algorithm: Optional[DynamicProgrammingAlgorithms | str] = None,
        seed: Optional[int | None] = None,
        **kwargs,
    ):
        """
        :param environment: internally assumed to be based on Board
        :param policy: a dictionary that maps a state to an action
        :param algorithm: an algorithm that will be used to find a policy
        (must be either PolicyIteration or ValueIteration)
        :param seed: is used to set a PRNG seed
        :param kwargs: arguments to pass to a method that calculates the policy
        """

        super().__init__(
            environment.observation_space,
            environment.action_space,
            policy=policy,
            seed=seed,
        )
        self.environment = environment
        self._unwrapped_env: Board = cast(Board, self.environment.unwrapped)

        self.P = self._unwrapped_env.P
        self.R = self._unwrapped_env.R

        self.dp_functions = {
            self.DynamicProgrammingAlgorithms.PolicyIteration: self.__policy_iteration,
            self.DynamicProgrammingAlgorithms.ValueIteration: self.__value_iteration,
        }

        if algorithm:
            assert isinstance(
                algorithm, (self.DynamicProgrammingAlgorithms, str)
            ), f"TypeError: algorithm must be of type DynamicProgrammingAlgorithms or str. Got: {type(algorithm)}"
            assert (
                algorithm in self.__class__.DynamicProgrammingAlgorithms
            ), f"ValueError: algorithm must be one of {self.DynamicProgrammingAlgorithms.__members__.keys()}"

            self._policy = self.dp_functions[algorithm](**kwargs)

    def find_policy(
        self, algorithm: Union[DynamicProgrammingAlgorithms, str], **kwargs
    ) -> Policy:
        self._policy = self.dp_functions[algorithm](**kwargs)
        return self._policy

    # Policy Iteration methods
    def __policy_evaluation(
        self, discount=0.1, stopping_criterion: float = 0.001
    ) -> Dict[Tuple[int, int, int, int], float]:
        """
        Uses an iterative algorithm to evaluate the value (v) of each state.
        Assumes a deterministic policy.

        @param discount: Determines how much the agent values future rewards over the immediate rewards
        @param stopping_criterion: The threshold that determines when the algorithm stops
        @return: A dictionary mapping each observation in the environment to an associated value
        """
        # Create a dictionary where the keys are all possible states of the environment
        all_v = {key: 0 for key in self.get_keys(self._obs_space)}

        # Relates to the accuracy of our V. Is compared to stopping_criterion
        delta = np.inf

        # Refrain all the values in the dictionary, until the difference between iterations is larger
        # than the threshold for the accuracy
        while delta >= stopping_criterion:
            delta = 0
            # The dictionary for the updated prediction for all state values
            new_all_v = all_v.copy()

            for state, action_sampler in self.items():
                # If we are in a terminal state, it's value is always 0
                if not self.P[state, action_sampler.sample()]:
                    new_all_v[state] = 0
                    continue

                v = all_v[state]
                ps = action_sampler.probabilities

                new_v = sum(tuple(ps[action] * sum(
                    tuple(
                        (1 / len(self.P[state, action])) * (self.R[new_state] + discount * all_v[new_state])
                        for new_state in self.P[state, action]
                    )) for action in ps.keys()))

                delta = max(delta, abs(new_v - v))
                new_all_v[state] = new_v
            all_v = new_all_v

        return all_v

    def __policy_improvement(
        self, value_dict: Dict[Tuple[int, int, int, int], float], discount: float
    ) -> tuple["DynamicProgrammingPolicy", bool]:
        """
        Finds a new policy based on the value of the current policy by acting greedily.

        @param discount: The discount used when solving the Bellman equations.
        @param stopping_criterion: Stopping criterion used for stopping when evaluating policy.
        @return: A tuple of the new policy, and if it changed with regard to the current policy.
        """
        # Initialise new policy, so old policy is not modified. Nice to keep current policy immutable.
        policy = DynamicProgrammingPolicy(self.environment)
        made_changes = False

        # Try to improve all states
        for state in self:
            # State is terminal
            if len(self.P[(state, 0)]) == 0:
                continue

            values = {}

            # Compute the expected value for all actions
            for action in self._all_actions:
                action_value = sum(
                    map(lambda new_state: self.R[new_state] + value_dict[new_state] * discount, # Get value for each successor
                        self.P[(state, action)],  # Get all possible successors for action
                    )) / len(self.P[(state, action)])  # Compute average value for action

                values[action] = action_value

            # Argmax
            new_action = max(values.items(), key=lambda kv: kv[1])[0]

            # Check if changes were made and change policy
            made_changes = new_action != self.get_action(state) or made_changes
            policy[state] = new_action

        return policy, made_changes

    def __policy_iteration(
        self, discount: float = 0.1, stopping_criterion: float = 0.001
    ) -> "DynamicProgrammingPolicy":
        """
        Performs policy iteration.

        @param discount: The discount used when solving the Bellman equations.
        @param evaluation_stopping_criterion: The stopping criterion used when evaluating the policy.
        @return: The optimised policy.
        """
        made_changes, policy = True, self

        # Loop while policy has not converged
        while made_changes:
            v = policy.__policy_evaluation(
                discount=discount, stopping_criterion=stopping_criterion
            )

            policy, made_changes = policy.__policy_improvement(v, discount=discount)
        return policy

    # Value iteration methods
    def __value_iteration(self, discount=0.1, stopping_criterion=0.001) -> "Policy":
        """
        Performs value iteration, computing an approximation of the optimal policy.

        @param discount: The discount factor used when solving the Bellman equations.
        @param stopping_criterion: Stopping criterion used for stopping when evaluating policy.
        @return: A new policy that is an approximation of the optimal policy.
        """

        # Create the new policy.
        policy = Policy(self._obs_space, self._act_space)
        # Create a dictionary where the keys are all possible states of the environment
        all_v = {key: 0 for key in self.get_keys(self._obs_space)}

        # Relates to the accuracy of our V. Is compared to stopping_criterion
        delta = np.inf

        while delta >= stopping_criterion:
            delta = 0

            new_all_v = deepcopy(all_v)

            for state in policy.keys():

                best_action = None
                best_val = -np.inf

                for action in self._all_actions:

                    if not self.P[state, action]:
                        new_all_v[state] = 0
                        continue

                    state_chance = 1 / len(self.P[state, action])

                    new_v = sum(
                        tuple(state_chance* (self.R[new_state] + discount * all_v[new_state]) for new_state in self.P[state, action])
                    )

                    if new_v > best_val:
                        best_val = new_v
                        best_action = action

                if best_action is not None:
                    new_all_v[state] = best_val
                    policy[state] = best_action
                    delta = max(delta, abs(best_val - all_v[state]))

            all_v = new_all_v

        return policy
