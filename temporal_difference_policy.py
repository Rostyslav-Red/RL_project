from policy import Policy
import gymnasium as gym
from typing import Callable
import numpy as np


def epsilon_greedy(q: dict[int, float], seed: int = None, epsilon: float = 0.5) -> int:
    rng = np.random.default_rng(seed)
    val = rng.uniform(0, 1)
    return max(q.items(), key=lambda kv: kv[1])[0] if val > epsilon else rng.choice(tuple(q.keys()))


class TemporalDifferencePolicy(Policy):

    def __init__(self, obs_space: gym.Space, act_space: gym.Space):
        super().__init__(obs_space, act_space)

    def sarsa(self,
              env: gym.Env,
              n_episodes: int,
              alpha: float = 0.5,
              gamma: float = 0.1,
              policy_func: Callable[[dict[int, float]], int] = epsilon_greedy
              ) -> Policy:
        """
        Uses the State-Action-Reward-State-Action (SARSA) to update the value function over the course of a number
        of episodes.

        @param env: Environment for which the algorithm should learn the policy.
        @param n_episodes: Number of episodes the algorithm should execute before convergence is assumed.
        @param alpha: Learning rate.
        @param gamma: Discount.
        @param policy_func: Function mapping a dictionary of actions mapping to values to an action.
        @return: A policy constructed based on the value function, computed from it using policy_func.
        """
        assert env.action_space == self._act_space and env.observation_space == self._obs_space, \
            "Environment does not match provided action or observation space."

        # Initialise Q(S, A)
        q = {(key, action): 0.0 for key in self for action in self._all_actions }

        for episode_n in range(n_episodes):
            obs, _ = env.reset()
            obs = self._obs_to_tuple(obs)

            action = policy_func(self._action_to_value(q, obs))
            terminal, truncated = False, False

            while not (terminal or truncated):
                new_obs, reward, terminal, truncated, _ = env.step(action)
                new_obs = self._obs_to_tuple(new_obs)

                new_action = policy_func(self._action_to_value(q, obs))

                # Update q
                q[obs, action] += alpha * (reward + gamma * q[new_obs, new_action] - q[obs, action])

                action, obs = new_action, new_obs

        return self._greedy_policy_from_q(q)


    def q_learning(self,
              env: gym.Env,
              n_episodes: int,
              alpha: float = 0.5,
              gamma: float = 0.1,
              policy_func: Callable[[dict[int, float]], int] = epsilon_greedy
              ) -> Policy:

        assert env.action_space == self._act_space and env.observation_space == self._obs_space, \
            "Environment does not match provided action or observation space."

        q = {(key, action): 0.0 for key in self for action in self._all_actions }

        for episode_n in range(n_episodes):
            obs, _ = env.reset()
            obs = self._obs_to_tuple(obs)

            while True:

                action = policy_func(self._action_to_value(q, obs))

                new_obs, reward, terminal, truncated, _ = env.step(action)
                new_obs = self._obs_to_tuple(new_obs)

                max_q_value = max(self._action_to_value(q, new_obs).values())
                q[obs, action] += alpha * (reward + gamma * max_q_value - q[obs, action])

                obs = new_obs

                if terminal or truncated:
                    break

        return self._greedy_policy_from_q(q)
