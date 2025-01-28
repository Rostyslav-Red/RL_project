from typing import Tuple, Optional
from constants import *


class Cell:
    """
    Class Cell represents a fundamental unit of the environment and saves as a basis for the Board
    """

    def __init__(
        self,
        position: Tuple[int, int],
        cell_type: int = 0,
        walls: Tuple[int, int, int, int] = (0, 0, 0, 0),
        holds_agent: bool = False,
    ):
        # The location of the cell in the format of (row, column)
        assert (
            isinstance(position, tuple)
            and len(position) == 2
            and all(isinstance(i, int) for i in position)
        ), "position must be a tuple of length 2 containing integers"
        self._position: Tuple[int, int] = position

        # The type of the cell, described in constants.cell_types
        assert (
            isinstance(cell_type, int) and cell_type in cell_types
        ), f"cell_type must be an int in range {min(cell_types)}-{max(cell_types)}"
        self._cell_type: int = cell_type

        """
        Represents whether there is a wall (1) or there is no wall(0) to the 
        left, right, top, and bottom of the cell respectively (e.g., (0, 1, 0, 1) would represent 
        a cell with a wall on the right and bottom, but not on the left or top).
        """
        assert (
            isinstance(walls, tuple)
            and len(walls) == 4
            and all(isinstance(i, int) and i in (0, 1) for i in walls)
        ), "walls must be a tuple of length 4 containing ints with values 0 or 1"
        self._walls: Tuple[int, int, int, int] = walls

        # True if the cat is on this cell, False otherwise
        assert isinstance(holds_agent, bool), "holds_agent must be a bool"
        self._holds_agent: bool = holds_agent

    # dunder methods
    def __str__(self) -> str:
        """
        0 -> empty -> E
        1 -> wall -> W
        2 -> fish -> F
        :return: None
        """
        return cell_types[self._cell_type][0].upper()

    # getters
    @property
    def position(self) -> Tuple[int, int]:
        return self._position

    @property
    def cell_type(self) -> int:
        return self._cell_type

    @property
    def walls(self) -> Tuple[int, int, int, int]:
        return self._walls

    @property
    def holds_agent(self) -> bool:
        return self._holds_agent

    # setters
    @cell_type.setter
    def cell_type(self, cell_type: int) -> None:
        assert (
            isinstance(cell_type, int) and cell_type in cell_types
        ), f"cell_type must be an int in range {min(cell_types)}-{max(cell_types)}"
        self._cell_type = cell_type

    @holds_agent.setter
    def holds_agent(self, holds_agent: bool) -> None:
        self._holds_agent = holds_agent


c = Cell((0, 0), 4, (0, 0, 0, 1))
print(c.cell_type)
c.cell_type = 4
print(c.walls)
print(c)
