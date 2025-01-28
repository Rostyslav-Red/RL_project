from board import Board
from constants import *
from cell import Cell
from copy import deepcopy

# An empty board of the board_size
empty_board = [
    [Cell(position=(j, i)) for i in range(board_size[1])] for j in range(board_size[0])
]

# A board from the example in the lecture
example_board = deepcopy(empty_board)
example_board[4][0].holds_agent = True
example_board[3][1].cell_type = 1
example_board[1][2].cell_type = 1

example_board[3][2].walls = (0, 1, 0, 0)
example_board[4][2].walls = (0, 1, 0, 0)
example_board[5][2].walls = (0, 1, 0, 0)

example_board[3][3].walls = (1, 0, 0, 0)
example_board[4][3].walls = (1, 0, 0, 0)
example_board[5][3].walls = (1, 0, 0, 0)

example_board[1][4].walls = (0, 0, 0, 1)
example_board[1][5].walls = (0, 0, 0, 1)

example_board[2][4].walls = (0, 0, 1, 0)
example_board[2][5].walls = (0, 0, 1, 0)

example_board[4][5].cell_type = 2

b = Board(example_board)

while True:
    print(b)
    direction = input("Where would you like to go?\n:\t")
    b.move(direction)
