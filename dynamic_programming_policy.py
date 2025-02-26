from policy import Policy
import gymnasium as gym
from typing import Dict, Tuple, Optional
import numpy as np


class DynamicProgrammingPolicy(Policy):

    def __init__(self,
                 environment: gym.Env,
                 policy: Optional[dict] = None,
                 seed: Optional[int | None] = None
        ):
        super().__init__(environment.observation_space, environment.action_space, policy=policy, seed=seed)
        self.environment = environment

        self._transition_function = self.environment.env.env.P
        self._reward_function = self.environment.env.env.R

    def __policy_evaluation(
            self, discount=0.1, stopping_criterion: float = 0.001
    ) -> Dict[Tuple[int, int, int, int], float]:
        """
        Uses an iterative algorithm to evaluate the value (v) of each state

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

            for state, action in self.items():
                # If we are in a terminal state, it's value is always 0
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

    def value_iteration(self, discount=0.1, stopping_criterion=0.001) -> 'Policy':
        """
        Performs value iteration, computing an approximation of the optimal policy.

        @param discount: The discount factor used when solving the Bellman equations.
        @param stopping_criterion: The stopping criterion used, if the difference is lower than this value, value
                                   iteration is said to have found the optimal value function.
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

            new_all_v = all_v.copy()

            for state in policy.keys():

                best_action = None
                best_val = float('-inf')

                for action in self._all_actions:

                    if not self._transition_function.get((state, action)):
                        new_all_v[state] = 0
                        continue

                    state_chance = 1 / len(self._transition_function[state, action])

                    new_v = sum(
                        tuple(
                            state_chance * (self._reward_function[new_state] + discount * all_v[new_state])
                            for new_state in self._transition_function[state, action]
                        )
                    )

                    if new_v > best_val:
                        best_val = new_v
                        best_action = action

                    # delta = max(delta, abs(best_val - all_v[state]))
                    # new_all_v[state] = best_val
                    #
                    # if best_action is not None:
                    #     policy[state] = best_action

                    if best_action is not None:
                        new_all_v[state] = best_val
                        policy[state] = best_action
                        delta = max(delta, abs(best_val - all_v[state]))

                all_v = new_all_v

        return policy

    def __improve(self, discount: float, stopping_criterion: float) -> tuple['DynamicProgrammingPolicy', bool]:
        """
        Performs one step of policy improvement.

        @param discount: The discount used when solving the Bellman equations.
        @param stopping_criterion: Stopping criterion used for stopping when evaluating policy.
        @return: A tuple of the new policy, and if it changed with regard to the current policy.
        """
        # Initialise new policy, so old policy is not modified. Nice to keep current policy immutable.
        policy = DynamicProgrammingPolicy(self.environment)
        made_changes = False
        transition_function = self.environment.env.env.P

        value_dict = self.__policy_evaluation(discount=discount, stopping_criterion=stopping_criterion)

        # Try to improve all states
        for state in self:
            # State is terminal
            if len(transition_function[(state, 0)]) == 0:
                continue

            values = {}

            # Compute the expected value for all actions
            for action in self._all_actions:
                action_value = sum(
                    map(
                        lambda new_state: self._reward_function[new_state] + value_dict[new_state] * discount,
                        # Get value for each successor
                        transition_function[(state, action)]  # Get all possible successors for action
                    )
                ) / len(transition_function[(state, action)])  # Compute average value for action

                values[action] = action_value

            # Argmax
            new_action = max(values.items(), key=lambda kv: kv[1])[0]

            # Check if changes were made and change policy
            made_changes = new_action != self[state] or made_changes
            policy[state] = new_action

        return policy, made_changes

    def policy_iterate(self, discount: float = 0.1, evaluation_stopping_criterion: float = 0.001) -> 'DynamicProgrammingPolicy':
        """
        Performs policy iteration.

        @param discount: The discount used when solving the Bellman equations.
        @param evaluation_stopping_criterion: The stopping criterion used when evaluating the policy.
        @return: The optimised policy.
        """
        made_changes, policy = True, self

        # Loop while policy has not converged
        while made_changes:
            policy, made_changes = policy.__improve(discount=discount, stopping_criterion=evaluation_stopping_criterion)
        return policy