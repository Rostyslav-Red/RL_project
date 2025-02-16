from agent import RandomAgent, Agent, HumanAgent
from board import Board, ConfiguredBoard
from constants import *
from cell import Cell
from copy import deepcopy
import numpy as np
import gymnasium as gym


if __name__ == "__main__":
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    options = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}

    board = gym.make("Board-v0")
    obs, _ = board.reset(options=options)

    # Possible agents: HumanAgent, RandomAgent
    agent = RandomAgent(board)
    print(f"Obtained reward: {agent.run_agent(obs)}")
