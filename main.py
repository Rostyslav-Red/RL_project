from agent import RandomAgent, HumanAgent, PolicyAgent
import numpy as np
import gymnasium as gym
from policy import Policy

if __name__ == "__main__":
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    options = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}

    board = gym.make("Board-v0")
    obs, _ = board.reset(seed=1)

    # Possible agents: HumanAgent, RandomAgent, PolicyAgent
    agent = PolicyAgent(board, Policy(board, seed=1))
    print(f"Obtained reward: {agent.run_agent(obs)}")
