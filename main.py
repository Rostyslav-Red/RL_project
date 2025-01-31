from board import Board
from constants import *
from cell import Cell
from copy import deepcopy
import numpy as np

# An empty board of the board_size
empty_board = [
    [Cell(position=(j, i)) for i in range(board_size[1])] for j in range(board_size[0])
]

# A board from the example in the lecture
example_board = deepcopy(empty_board)

# Define a cell that holds an agent
example_board[0][0].holds_agent = True

# Define a cell that holds a target
example_board[3][3].holds_target = True

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

b = Board(example_board)

while True:
    print(b)
    direction = input("Where would you like to go?\n:\t")
    match direction:
        case "right":
            b.move(np.array((0, 1)))
        case "left":
            b.move(np.array((0, -1)))
        case "up":
            b.move(np.array((-1, 0)))
        case "down":
            b.move(np.array((1, 0)))
        case _:
            b.move(np.array((0, 0)))
