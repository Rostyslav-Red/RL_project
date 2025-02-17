from agent import RandomAgent, Agent, HumanAgent, PolicyAgent
import numpy as np
import gymnasium as gym

from policy import Policy

if __name__ == "__main__":
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    options = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}

    board = gym.make("Board-v0")
    obs, _ = board.reset(options=options, seed=0)

    # Possible agents: HumanAgent, RandomAgent
    agent = PolicyAgent(board, Policy(board, seed=0))
    print(f"Obtained reward: {agent.run_agent(obs)}")
