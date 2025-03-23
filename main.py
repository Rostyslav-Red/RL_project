from agent import RandomAgent, HumanAgent, PolicyAgent
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from dynamic_programming_policy import DynamicProgrammingPolicy
from policy import Policy
from temporal_difference_policy import TemporalDifferencePolicy

if __name__ == "__main__":
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    options = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}

    board = gym.make("Board-v0", render_mode="human")

    # Checks if board is a valid environment
    # print(check_env(board.unwrapped))

    ### Block for computing policy, only run when computing a new policy
    # p = TemporalDifferencePolicy(board.observation_space, board.action_space).q_learning(board, n_episodes=100, alpha=0.1, gamma=0.9)
    # p.save("policies/td_qlearning.json")
    ###

    obs, _ = board.reset(options=options, seed=100)

    # p = Policy.load(board, "policies/value_iteration_policy.json")
    # p = DynamicProgrammingPolicy(
    #    board, algorithm="PolicyIteration", discount=0.1, stopping_criterion=0.00000001
    # )

    """
    Methods of choosing a dynamic policy:
    
    1. By loading a saved policy
        p = Policy.load(board, "policies/value_iteration_policy.json")

    2. By specifying the 'algorithm' argument of the instance of DynamicProgrammingPolicy
        p = DynamicProgrammingPolicy(
            board, algorithm="ValueIteration", discount=0.1, stopping_criterion=20
        )
    
    3. By calling the 'find_policy' method on the instance of DynamicProgrammingPolicy
        p = DynamicProgrammingPolicy(board).find_policy("ValueIteration")
    """
    p = Policy.load(board, "policies/td_qlearning.json")

    # Possible agents: HumanAgent, RandomAgent, PolicyAgent
    agent = PolicyAgent(board, p)

    print(f"Obtained reward: {agent.run_agent(obs)}")
