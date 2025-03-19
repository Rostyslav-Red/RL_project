from agent import RandomAgent, HumanAgent, PolicyAgent
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from dynamic_programming_policy import DynamicProgrammingPolicy
from policy import Policy
from temporal_difference_policy import TemporalDifferencePolicy
from monte_carlo_policy import MonteCarloPolicy

if __name__ == "__main__":
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    options = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}

    board = gym.make("Board-v0")

    # Checks if board is a valid environment
    # print(check_env(board.unwrapped))

    ### Block for computing policy, only run when computing a new policy
    # p = TemporalDifferencePolicy(board.observation_space, board.action_space).sarsa(board, n_episodes=1000)
    # p.save("policies/td_sarsa.json")
    p = MonteCarloPolicy(board).policy_evaluation(board)
    ###

    #obs, _ = board.reset(options=options, seed=100)

    #p = Policy.load(board, "policies/td_sarsa.json")
    # Possible agents: HumanAgent, RandomAgent, PolicyAgent
    #agent = PolicyAgent(board, p)

    #print(f"Obtained reward: {agent.run_agent(obs)}")
