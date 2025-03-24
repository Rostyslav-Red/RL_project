from policy import Policy
import gymnasium as gym
from typing import List, Tuple
import numpy as np


class MonteCarloPolicy(Policy):

    def __init__(self, environment: gym.Env, policy: dict = None):
        super().__init__(environment.observation_space, environment.action_space, policy=policy)
        self.environment = environment

    def first_visit(self, episode: List[Tuple[Tuple[gym.Space, int], float]]):
        all_v = {}
        all_q = {}
        reward = 0
        episode.reverse()
        for i in range(len(episode)):
            reward = 0.1*reward+episode[i][1]
            if episode[i][0][0] not in list(all_v.keys()):
                all_v[episode[i][0][0]] = reward
            if episode[i][0] not in list(all_q.keys()):
                all_q[episode[i][0]] = reward

        return all_v, all_q

    def policy_evaluation(self, env: gym.Env):
        policy = Policy(self._obs_space, self._act_space)

        # should call all posible states as an env so i can call the posistions of the cat
        # and mouse so i can control the reset and it isn't random
        states = self.keys() # these are tuples

        all_v = {}
        all_q = {}
        # runs episodes for each posible action for each posible state
        for state in states:
            # print("state")
            for action in self._all_actions:
                # print("action")
                obs, _ = env.reset()
                episode = []
                while True:
                    new_obs, reward, terminal, truncated, _ = env.step(action)
                    new_obs = self._obs_to_tuple(new_obs)

                    if terminal or truncated:
                        break

                    new_action = policy.get_action(new_obs)

                    # Update episode
                    episode.append(((obs, action), float(reward)))

                    action, obs = new_action, new_obs

                # does first visit for bout v and q on an episode
                episode_v, episode_q = self.first_visit(episode)

                # adds all values of one episode to a big dictionary
                for k, v in episode_v.items():
                    if k in all_v.keys():
                        all_v[k].append(v)
                    else:
                        all_v[k] = [v]
                for k, v in episode_q.items():
                    if k in all_q.keys():
                        all_q[k].append(v)
                    else:
                        all_q[k] = [v]

        # gets the average for each value in a dictionary getting the proper reward for each v and q
        avr_v = {}
        avr_q = {}
        for k, v in all_v.items():
            avr_v[k] = np.mean(v)
        for k, v in all_q.items():
            avr_q[k] = np.mean(v)

        print(avr_v[list(avr_v.keys())[10]])





