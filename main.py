from agent import RandomAgent, Agent, HumanAgent
from board import Board
from constants import *
from cell import Cell
from copy import deepcopy
import numpy as np


if __name__ == "__main__":
    # An empty board of the board_size
    empty_board = [
        [Cell(position=(j, i)) for i in range(board_size[1])] for j in range(board_size[0])
    ]

    # A board from the example in the lecture
    example_board = deepcopy(empty_board)

    # Define a cell that holds an agent
    options = {
        "cat_position": np.array([0, 0]),
        "target_position": np.array([3, 3])
    }

    # Define the locations of the trees
    example_board[0][1].cell_type = 1
    example_board[0][3].cell_type = 1
    example_board[2][0].cell_type = 1
    example_board[2][3].cell_type = 1

    # Define the locations of the walls
    example_board[0][0].walls = np.array([0, 1])
    example_board[0][1].walls = np.array([0, -1])
    example_board[0][2].walls = np.array([0, 1])
    example_board[0][3].walls = np.array([0, -1])
    example_board[1][2].walls = np.array([1, 0])
    example_board[2][2].walls = np.array([-1, 0])
    example_board[2][0].walls = np.array([0, 1])
    example_board[2][1].walls = np.array([1, -1])
    example_board[3][1].walls = np.array([-1, 0])
    example_board[2][3].walls = np.array([1, 0])
    example_board[3][2].walls = np.array([0, 1])
    example_board[3][3].walls = np.array([-1, -1])

    b, obs, _ = Board.board_factory(example_board, options=options)

    # Possible agents: HumanAgent, RandomAgent
    agent = RandomAgent(b)
    print(f"Obtained reward: {agent.run_agent(obs)}")
