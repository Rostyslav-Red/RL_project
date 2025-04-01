from agent import RandomAgent, HumanAgent, PolicyAgent
from deep_qlearning import DeepQLearningAgent, get_data_and_train, RLData
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
        
        
    Methods of creating a Deep Q-Learning agent:
    1. By loading saved model weights
        agent = DeepQLearningAgent.load(board, "path-to-weights.pt")
    
    2. By training a new model on a dataset. Data can be loaded from sampled data. Note there are many customisation 
       options for training, read the documentation for those.
        data = RLData.load("policies/episodes.json")
        agent = DeepQLearningAgent.build_model(board, (hidden_layer1, hidden_layer2, ...))
        agent.train(n_epochs=10, data=data, **kwargs)
        
    3. By training a new model on a newly sampled dataset. Note there are many customisation options for training, 
       read the documentation for those.
        agent = get_data_and_train(board, (hidden_layer1, hidden_layer2, ...), **kwargs)
    """

    # Possible agents: HumanAgent, RandomAgent, PolicyAgent, DeepQLearningAgent
    agent = DeepQLearningAgent.load(board, "policies/model.pt")


    print(f"Obtained reward: {agent.run_agent(obs)}")
