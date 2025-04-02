from policy import Policy
import gymnasium as gym
from typing import List, Tuple, Optional
from action_sampler import ActionSampler
from gymnasium.core import ActType, ObsType
from enum import Enum


class MonteCarloPolicy(Policy):
    """
    Implements On-Policy MC Prediction and Control without exploring starts, using First-Visit Monte-Carlo prediction
    and epsilon-greedy Monte-Carlo control.
    """
    class MonteCarloAlgorithms(str, Enum):
        FIRST_VISIT_EPSILON_GREEDY = "FirstVisitEpsilonGreedy"

    def __init__(self,
                 obs_space: ObsType,
                 act_space: ActType,
                 policy: dict = None,
                 algorithm: Optional[MonteCarloAlgorithms | str] = None,
                 **kwargs):
        """
        Constructs the MonteCarloPolicy.

        @param obs_space: The observation space of the environment this policy will describe.
        @param act_space: The action space of the environment this policy will describe.
        @param policy: Optional, the starting policy.
        """
        super().__init__(obs_space, act_space, policy=policy)
        self.__all_q, self.__returns = {}, {}
        self.__reset()

        self.mc_functions = {
            self.MonteCarloAlgorithms.FIRST_VISIT_EPSILON_GREEDY: self.first_visit_monte_carlo_control,
        }

        if algorithm:
            assert isinstance(
                algorithm, (self.MonteCarloAlgorithms, str)
            ), f"TypeError: algorithm must be of type DynamicProgrammingAlgorithms or str. Got: {type(algorithm)}"
            assert (
                algorithm in self.__class__.MonteCarloAlgorithms
            ), f"ValueError: algorithm must be one of {self.MonteCarloAlgorithms.__members__.keys()}"

            self.mc_functions[algorithm](**kwargs)

    def first_visit_monte_carlo_control(self,
                                        env: gym.Env,
                                        n_episodes: int = 100,
                                        gamma: float = 0.5,
                                        epsilon: float = 0.5,
                                        reset: bool = True
                                        ) -> 'MonteCarloPolicy':
        """
        Finds the policy using Monte Carlo method in n episodes.

        @param env: Environment for which the policy is found.
        @param n_episodes: Number of episodes to run the Monte Carlo method.
        @param gamma: Discount factor.
        @param epsilon: The probability the optimal action will be picked.
        @param reset: Should q and returns be reinitialised? Default is True.
        @return: Found policy.
        """
        if reset:
            self.__reset()

        for i in range(n_episodes):
            obs, _ = env.reset()
            obs = self._obs_to_tuple(obs)
            episode = []

            # Runs episode
            while True:
                action = self.get_action(obs)

                new_obs, reward, terminal, truncated, _ = env.step(action)
                new_obs = self._obs_to_tuple(new_obs)

                if terminal or truncated:
                    break

                # Update episode
                new = ((obs, action), float(reward))
                episode.append(new)

                obs = new_obs

            # Does first visit for q on an episode
            self.__first_visit(episode, gamma, epsilon)

        return self

    def __first_visit(self,
                      episode: List[Tuple[Tuple[Tuple[int, ...], int], float]],
                      discount: float,
                      epsilon: float
                      ) -> None:
        """
        First-visit Monte Carlo prediction.

        @param episode: Episode to predict.
        @param discount: Discount factor.
        @param epsilon: The probability the optimal action will be picked.
        """
        reward = 0
        episode.reverse()

        state_actions = tuple(map(lambda step: step[0], episode))

        for i, step in enumerate(episode):
            reward = discount * reward + step[1]  # Update returns

            # Make sure only first visit counts
            if step[0] not in state_actions[i + 1:]:
                self.__returns[step[0]] += 1  # Count how many returns for this state, action we have
                self.__all_q[step[0]] += reward / self.__returns[step[0]]  # Take average

                # Update policy using newly acquired knowledge
                state = step[0][0]
                action_values = self._action_to_value(self.__all_q, state)
                best_action = max(action_values, key=action_values.get)
                self[state] = ActionSampler.epsilon_greedy_action_sampler(self._all_actions, best_action, epsilon)

    def __reset(self) -> None:
        """
        Resets the q function and the returns dictionary to their base value.
        """
        self.__all_q = {(state, action): 0.0 for state in self.get_keys(self._obs_space) for action in
                        self._all_actions}
        self.__returns = {key: 0 for key in self.__all_q.keys()}
