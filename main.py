from agent import RandomAgent, HumanAgent, PolicyAgent, DeepQLearningAgent
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from dynamic_programming_policy import DynamicProgrammingPolicy
from policy import Policy
from temporal_difference_policy import TemporalDifferencePolicy
import torch

if __name__ == "__main__":
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    options = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}

    board = gym.make("Board-v0", render_mode=None)

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
    agent = DeepQLearningAgent.build_model(board, (10, 30, 10))

    model = agent.model

    # for observation in Policy.get_keys(board.observation_space):
    #     print(observation)
    #     print(model.forward((torch.tensor(tuple(map(float, observation))).to("cuda"))))

    # agent.train(board, 100, retarget=1)
    # agent.save("policies/model.pt")
    agent = DeepQLearningAgent.load(board, "policies/model.pt")

    for observation in Policy.get_keys(board.observation_space):
        print(observation)
        print(model.forward((torch.tensor(tuple(map(float, observation))).to("cuda"))))

    print(f"Obtained reward: {agent.run_agent(obs)}")
