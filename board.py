from typing import List, Tuple
from unittest import case

from cell import Cell
from constants import *
from copy import deepcopy
from itertools import chain


class Board:
    def __init__(self, board: List[List[Cell]]):
        self._board: List[List[Cell]] = board
        self._cat_position: Tuple[int] = (0,)
        for cell in list(chain.from_iterable(board)):
            if cell.holds_agent:
                self._cat_position = cell.position
                break
        else:
            raise ValueError("No cat found on the board")
        self._board_size: Tuple[int, int] = (len(board), len(board[0]))

    # dunder methods
    def __str__(self):
        horizontal_line = "  " + "—   " * board_size[1]
        result = horizontal_line
        for row in self._board:
            char_line = "| "
            wall_line = "| "
            for cell in row:
                char_line += str(cell) + (" | " if cell.walls[1] else "   ")
                wall_line += "—   " if cell.walls[3] else "    "
            char_line = char_line[:-2]
            wall_line = wall_line[:-2]
            char_line += " |"
            wall_line += " |"
            result += "\n" + char_line + "\n" + wall_line
        result = result[: -(self._board_size[1] * 4 + 3)]
        result += "\n" + horizontal_line

        return result

    # getters
    @property
    def board(self):
        return self._board

    @property
    def cat_position(self):
        return self._cat_position

    # setters
    @cat_position.setter
    def cat_position(self, cat_position: Tuple[int]):
        self._cat_position = cat_position

    # movement
    def move(self, direction: str):
        match direction:
            case "right":
                self._move_right()
            case "left":
                self._move_left()
            case "up":
                self._move_up()
            case "down":
                self._move_down()
            case _:
                raise ValueError("Invalid direction")

    def _move_right(self):
        target_location = (self._cat_position[0], self._cat_position[1] + 1)
        if target_location[1] >= self._board_size[1]:
            return
        current_cell = self._board[self._cat_position[0]][self._cat_position[1]]
        if current_cell.walls[1]:
            return
        self._board[self._cat_position[0]][self._cat_position[1]].holds_agent = False
        self._board[target_location[0]][target_location[1]].holds_agent = True
        self._cat_position = target_location

    def _move_left(self):
        target_location = (self._cat_position[0], self._cat_position[1] - 1)
        if target_location[1] < 0:
            return
        current_cell = self._board[self._cat_position[0]][self._cat_position[1]]
        if current_cell.walls[0]:
            return
        self._board[self._cat_position[0]][self._cat_position[1]].holds_agent = False
        self._board[target_location[0]][target_location[1]].holds_agent = True
        self._cat_position = target_location

    def _move_up(self):
        target_location = (self._cat_position[0] - 1, self._cat_position[1])
        if target_location[0] < 0:
            return
        current_cell = self._board[self._cat_position[0]][self._cat_position[1]]
        if current_cell.walls[2]:
            return
        self._board[self._cat_position[0]][self._cat_position[1]].holds_agent = False
        self._board[target_location[0]][target_location[1]].holds_agent = True
        self._cat_position = target_location

    def _move_down(self):
        target_location = (self._cat_position[0] + 1, self._cat_position[1])
        if target_location[0] >= self._board_size[0]:
            return
        current_cell = self._board[self._cat_position[0]][self._cat_position[1]]
        if current_cell.walls[3]:
            return
        self._board[self._cat_position[0]][self._cat_position[1]].holds_agent = False
        self._board[target_location[0]][target_location[1]].holds_agent = True
        self._cat_position = target_location
