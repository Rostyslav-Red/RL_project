from agent import RandomAgent, HumanAgent, PolicyAgent
from deep_qlearning import DeepQLearningAgent, RLData
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from dynamic_programming_policy import DynamicProgrammingPolicy
from policy import Policy
from temporal_difference_policy import TemporalDifferencePolicy
import torch

if __name__ == "__main__":
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    options = {"cat_position": np.array([0, 2]), "target_position": np.array([2, 0])}

    board = gym.make("Board-v0", render_mode="human")

    # Checks if board is a valid environment
    # print(check_env(board.unwrapped))

    ### Block for computing policy, only run when computing a new policy
    # p = TemporalDifferencePolicy(board.observation_space, board.action_space).sarsa(board, n_episodes=1000)
    # p.save("policies/td_sarsa.json")
    ###

    obs, _ = board.reset()

    # p = Policy.load(board, "policies/value_iteration_policy.json")
    # p = DynamicProgrammingPolicy(
    #     board, algorithm="ValueIteration", discount=0.1, stopping_criterion=0.00000001
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

    # Possible agents: HumanAgent, RandomAgent, PolicyAgent

    # Generate episode data
    # data = RLData.sample_data(board, 1000)
    # data.save("policies/episodes.json")
    # data = RLData.load("policies/episodes.json")

    # Create Neural Network, train and save it
    # agent = DeepQLearningAgent.build_model(board, (10, 10))
    # agent.train(10, data, retarget=100, batch_size=1024)
    # agent.save("policies/model.pt")

    # Load DeepQLearningAgent from weights
    agent = DeepQLearningAgent.load(board, "policies/model.pt")


    print(f"Obtained reward: {agent.run_agent(obs)}")
