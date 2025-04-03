from agent import RandomAgent, HumanAgent, PolicyAgent
from deep_qlearning import DeepQLearningAgent, get_data_and_train, RLData
import numpy as np
import gymnasium as gym
from monte_carlo_policy import MonteCarloPolicy
from dynamic_programming_policy import DynamicProgrammingPolicy
from policy import Policy
from temporal_difference_policy import TemporalDifferencePolicy

if __name__ == "__main__":
    # Setup board
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    # Manually choose starting positions here, make sure positions are 2D vectors with values in 0 <= x <= 3.
    options = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}

    board = gym.make("Board-v0", render_mode="human")

    # Remove options here to randomise positions, add seed kwarg to run at the same random seed.
    obs, _ = board.reset(options=options)

    # Create policy here, following instructions found in README.md
    # Possible policies: Policy, DynamicProgrammingPolicy, MonteCarloPolicy, TemporalDifferencePolicy

    # policy = TemporalDifferencePolicy(
    #     board.observation_space,
    #     board.action_space,
    #     algorithm="QLearning",
    #     env=board,
    #     n_episodes=1000,
    #     alpha=0.5,
    #     gamma=0.9,
    # )

    policy = MonteCarloPolicy(
        board.observation_space, board.action_space
    ).first_visit_monte_carlo_control(
        env=board,
        n_episodes=100000,
        gamma=0.9,
        epsilon=0.1,
    )

    # Create agent here
    # Possible agents: HumanAgent, RandomAgent, PolicyAgent, DeepQLearningAgent
    agent = PolicyAgent(board, policy)
    # Run agent and print final reward.
    print(f"Obtained reward: {agent.run_agent(obs)}")
