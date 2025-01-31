from typing import List, Tuple, Annotated, Optional
from unittest import case

from cell import Cell
from constants import *
from copy import deepcopy
from itertools import chain
import numpy as np
import random


class Board:
    def __init__(self, board: List[List[Cell]]):
        self._board: List[List[Cell]] = board
        self._cat_position: Optional[Annotated[np.typing.NDArray[np.int_], (2,)]] = None
        self._target_position: Optional[Annotated[np.typing.NDArray[np.int_], (2,)]] = (
            None
        )

        for cell in list(chain.from_iterable(board)):
            if cell.holds_agent:
                self._cat_position: Annotated[np.typing.NDArray[np.int_], (2,)] = (
                    np.array(cell.position)
                )
            if cell.holds_target:
                self._target_position: Annotated[np.typing.NDArray[np.int_], (2,)] = (
                    np.array(cell.position)
                )
            if self._cat_position is not None and self._target_position is not None:
                break
        else:
            raise ValueError("No cat or target found on the board")
        self._board_size: Tuple[int, int] = (len(board), len(board[0]))

    # dunder methods
    def __str__(self):
        horizontal_line = " — " + "— — — " * board_size[1]
        result = horizontal_line
        for row in self._board:
            char_line = "|  "
            wall_line = "|  "
            for cell in row:
                char_line += str(cell) + ("  |  " if cell.walls[1] == 1 else "     ")
                wall_line += "—     " if cell.walls[0] == 1 else "      "
            char_line = char_line[:-4]
            wall_line = wall_line[:-4]
            char_line += "  |"
            wall_line += "  |"
            result += "\n" + char_line + "\n" + wall_line
        result = result[: -(self._board_size[1] * 6 + 3)]
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
    def move(self, direction: Annotated[np.typing.NDArray[np.int_], (2,)]):
        # moves the cat in the specified direction
        self._move(direction)

        # moves the mouse in a random allowed direction. If none are allowed, the mouse doesn't move
        possible_directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        random.shuffle(possible_directions)

        for target_direction in possible_directions:
            if self._move(np.array(target_direction), move_target=True):
                return True
        return False

    def _move(
        self,
        direction: Annotated[np.typing.NDArray[np.int_], (2,)],
        move_target: bool = False,
    ):
        assert direction.tolist() in [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
        ], "Invalid direction"

        """
        direction must be a numpy array with one of the following values: (1, 0), (-1, 0), (0, 1), (0, -1)
        The first value corresponds to the left-right direction, and the second value to the up-down direction.
        Specifically:
            - (0, 1) -> right
            - (0, -1) -> left
            - (1, 0) -> down
            - (-1, 0) -> up
        """

        if not move_target:
            location = self._cat_position
        else:
            location = self._target_position

        possible_location = location + direction

        # ensures that the cat and the mouse cannot move through the borders of the level
        if not 0 <= possible_location[0] < self._board_size[0]:
            return
        if not 0 <= possible_location[1] < self._board_size[1]:
            return

        current_cell = self._board[location[0]][location[1]]

        # ensures the cat cannot move through the walls withing the level
        if not move_target:
            ind = np.nonzero(direction)[0][0]
            if direction[ind] == current_cell.walls[ind]:
                return

        # if none of the checks fail, update the locations of a cat/mouse
        if not move_target:
            self._board[location[0]][location[1]].holds_agent = False
            self._board[possible_location[0]][possible_location[1]].holds_agent = True
            self._cat_position = possible_location
        else:
            self._board[location[0]][location[1]].holds_target = False
            self._board[possible_location[0]][possible_location[1]].holds_target = True
            self._target_position = possible_location
        return True
