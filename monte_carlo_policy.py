from policy import Policy
import gymnasium as gym
from typing import List, Tuple
import numpy as np
import action_sampler


class MonteCarloPolicy(Policy):

    def __init__(self, environment: gym.Env, policy: dict = None):
        super().__init__(environment.observation_space, environment.action_space, policy=policy)
        self.environment = environment
        self.policy = Policy(self._obs_space, self._act_space)
        self.all_v = {state: 0.0 for state in self.get_keys(self._obs_space)}
        self.all_q = {(state, action): 0.0 for state in self.get_keys(self._obs_space) for action in self._all_actions}

    def find(self, episode_n: int = 100, discount: float = 0.5, epsilon: float = 0.1):
        """
        Finds the policy using Monte Carlo method in n episodes.

        @param episode_n: Number of episodes to run the Monte Carlo method.
        @param discount: Discount factor.
        @param epsilon: Epsilon.
        @return:  Found policy.
        """
        env = self.environment
        for i in range(episode_n):
            obs, _ = env.reset()
            obs = self._obs_to_tuple(obs)
            episode = []

            # runs episode
            while True:
                action = self.policy.get_action(obs)

                new_obs, reward, terminal, truncated, _ = env.step(action)
                new_obs = self._obs_to_tuple(new_obs)

                if terminal or truncated:
                    break

                # Update episode
                new=((obs, action), float(reward))
                episode.append(new)

                obs = new_obs
            print('did episode:', i)

            # does first visit for bout v and q on an episode
            self.first_visit(episode, discount)
            print('did first visit:')
            self.policy_improvement(epsilon)
            print('did policy improvement')
        return self.policy

    def first_visit(self, episode: List[Tuple[Tuple[Tuple[int, ...], int], float]], discount):
        """
        First-visit Monte Carlo policy.

        @param episode: Episode to visit.
        @param discount: Discount factor.
        @return: None
        """
        all_v = {}
        all_q = {}
        reward = 0
        episode.reverse()
        for i in range(len(episode)):
            reward = discount * reward + episode[i][1]
            all_v[episode[i][0][0]] = reward
            all_q[episode[i][0]] = reward

        # adds all values of one episode to a big dictionary and gets the average for each entry
        for k, v in all_v.items():
            if self.all_v[k] != 0:
                self.all_v[k] = (self.all_v[k] + v)/2
            else:
                self.all_v[k] = v
        for k, v in all_q.items():
            if self.all_q[k] != 0:
                self.all_q[k] = (self.all_q[k]+v)/2
            else:
                self.all_q[k] = v
        return

    def policy_improvement(self, epsilon):
        """
        Monte Carlo policy improvement.

        @param epsilon: Epsilon.
        @return: None
        """
        for state in self.get_keys(self._obs_space):
            action_values = {action: self.all_q[(state, action)] for action in self._all_actions}
            best_action = max(action_values, key=action_values.get)
            self.policy[state].epsilon_greedy_action_sampler(self._all_actions, best_action, epsilon)
        return




