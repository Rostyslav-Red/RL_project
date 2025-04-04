from policy import Policy
import gymnasium as gym
from typing import Callable, Optional
import numpy as np
from enum import Enum
from action_sampler import ActionSampler


class TemporalDifferencePolicy(Policy):
    """
    Implementation of the Temporal Difference Learning algorithms. Both SARSA and Q-Learning are implemented here.
    """

    class TemporalDifferenceAlgorithms(str, Enum):
        SARSA = "SARSA"
        QLearning = "QLearning"

    def __init__(
        self,
        obs_space: gym.Space,
        act_space: gym.Space,
        *,
        seed: Optional[int] = None,
        algorithm: Optional[TemporalDifferenceAlgorithms | str] = None,
        **kwargs,
    ):
        """
        Constructs the TemporalDifferencePolicy

        @param obs_space: Observation space of the environment the policy describes.
        @param act_space: Action space of the environment the policy describes.
        @param algorithm: Optional, if given, calls specified algorithm with keyword arguments specified.
        @param kwargs: Arguments used in the algorithm call.
        """
        super().__init__(obs_space, act_space, seed=seed)

        self.__q = {}
        self.__reset()

        self.td_functions = {
            self.TemporalDifferenceAlgorithms.SARSA: self.sarsa,
            self.TemporalDifferenceAlgorithms.QLearning: self.q_learning,
        }

        if algorithm:
            assert isinstance(
                algorithm, (self.TemporalDifferenceAlgorithms, str)
            ), f"TypeError: algorithm must be of type DynamicProgrammingAlgorithms or str. Got: {type(algorithm)}"
            assert (
                algorithm in self.__class__.TemporalDifferenceAlgorithms
            ), f"ValueError: algorithm must be one of {self.TemporalDifferenceAlgorithms.__members__.keys()}"

            self._policy = self.td_functions[algorithm](**kwargs)

    def sarsa(
        self,
        env: gym.Env,
        n_episodes: int,
        alpha: float = 0.5,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        reset: bool = True,
    ) -> Policy:
        """
        Uses the State-Action-Reward-State-Action (SARSA) to update the value function over the course of a number
        of episodes.

        @param env: Environment for which the algorithm should learn the policy.
        @param n_episodes: Number of episodes the algorithm should execute before convergence is assumed.
        @param alpha: Learning rate.
        @param gamma: Discount.
        @param epsilon: Epsilon used for epsilon greedy decisions.
        @param reset: Should the state-action value function be reset at the start of the algorithm.
        @return: A policy constructed based on the value function, computed from it using policy_func.
        """
        assert (
            env.action_space == self._act_space
            and env.observation_space == self._obs_space
        ), "Environment does not match provided action or observation space."

        if reset:
            self.__reset()

        current_render_mode = env.render_mode
        if env.render_mode:
            env.unwrapped.render_mode = "None"

        for episode_n in range(n_episodes):
            obs, _ = env.reset(seed=int(self._rng.integers(0, 1000000)))
            obs = self._obs_to_tuple(obs)

            action = self._epsilon_greedy_action(self.__q, obs, epsilon).sample()
            terminal, truncated = False, False

            # Train over an episode
            while not (terminal or truncated):
                new_obs, reward, terminal, truncated, _ = env.step(action)
                new_obs = self._obs_to_tuple(new_obs)

                new_action = self._epsilon_greedy_action(self.__q, new_obs, epsilon).sample()

                # Update q
                self.__q[obs, action] += alpha * (
                    reward
                    + gamma * self.__q[new_obs, new_action]
                    - self.__q[obs, action]
                )

                action, obs = new_action, new_obs

        env.unwrapped.render_mode = current_render_mode

        # Compute greedy policy on value function.
        return self._greedy_policy_from_q(self.__q)

    def q_learning(
        self,
        env: gym.Env,
        n_episodes: int,
        alpha: float = 0.5,
        gamma: float = 0.9,
        epsilon: float = 0.3,
        reset: bool = True,
    ) -> Policy:
        """
        Uses the Q-Learning algorithm to update the value function over the course of a number
        of episodes.

        @param env: Environment for which the algorithm should learn the policy.
        @param n_episodes: Number of episodes the algorithm should execute before convergence is assumed.
        @param alpha: Learning rate.
        @param gamma: Discount.
        @param epsilon: Epsilon used for epsilon greedy decisions.
        @param reset: Should the state-action value function be reset at the start of the algorithm.
        @return: A policy constructed based on the value function, computed from it using policy_func.
        """

        assert (
            env.action_space == self._act_space
            and env.observation_space == self._obs_space
        ), "Environment does not match provided action or observation space."

        if reset:
            self.__reset()

        current_render_mode = env.render_mode
        if env.render_mode:
            env.unwrapped.render_mode = "None"

        for episode_n in range(n_episodes):
            obs, _ = env.reset(seed=int(self._rng.integers(0, 1000000)))
            obs = self._obs_to_tuple(obs)

            terminal, truncated = False, False

            # Learn over an episode
            while not (terminal or truncated):
                action = self._epsilon_greedy_action(self.__q, obs, epsilon).sample()

                new_obs, reward, terminal, truncated, _ = env.step(action)
                new_obs = self._obs_to_tuple(new_obs)

                max_q_value = max(self._action_to_value(self.__q, new_obs).values())
                self.__q[obs, action] += alpha * (
                    reward + gamma * max_q_value - self.__q[obs, action]
                )

                obs = new_obs

        env.unwrapped.render_mode = current_render_mode

        # Compute greedy policy on value function.
        return self._greedy_policy_from_q(self.__q)

    def __reset(self) -> None:
        """Resets the state-action value function back to 0 for all state-action pairs."""
        self.__q = {(key, action): 0.0 for key in self for action in self._all_actions}
