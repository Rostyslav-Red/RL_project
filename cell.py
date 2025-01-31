from typing import Tuple, Optional, Annotated
import numpy as np
from constants import *


class Cell:
    """
    Class Cell represents a fundamental unit of the environment and saves as a basis for the Board
    """

    def __init__(
        self,
        position: Tuple[int, int],
        cell_type: int = 0,
        walls: Annotated[np.typing.NDArray[np.int_], (2,)] = np.array([0, 0]),
        holds_agent: bool = False,
        holds_target: bool = False,
    ):
        # The location of the cell in the format of (row, column)
        assert (
            isinstance(position, tuple)
            and len(position) == 2
            and all(isinstance(i, int) for i in position)
        ), "position must be a tuple of length 2 containing integers"
        self._position: Tuple[int, int] = position

        # The type of the cell, described in constants.cell_types
        self._cell_type: int = cell_type

        """
        Represents the locations of the walls around the cell.
        The first digit corresponds to the down-top axis, and the second one to the left-right axis.
        Specifically:
        (0, -1) means that there is a wall to the left of the cell
        (0, 1) means that there is a wall to the right of the cell
        (-1, 0) means that there is a wall above the cell
        (1, 1) means that there are walls to the right and below the cell
        """
        self._walls: Annotated[np.typing.NDArray[np.int_], (2,)] = walls

        # True if the cat is on this cell, False otherwise
        self._holds_agent: bool = holds_agent

        # True if the mouse is on this cell, False otherwise
        self._holds_target: bool = holds_target

    # dunder methods
    def __str__(self) -> str:
        """
        0 -> empty -> ·
        1 -> hole -> ○
        :return: None
        """
        if self._holds_agent and self._holds_target:
            return "X"
        if self._holds_agent:
            return "C"
        if self._holds_target:
            return "M"

        return cell_representations[self._cell_type][0].upper()

    def __copy__(self) -> "Cell":
        return Cell(self._position, self._cell_type, self._walls, self._holds_agent)

    # getters
    @property
    def position(self) -> Tuple[int, int]:
        return self._position

    @property
    def cell_type(self) -> int:
        return self._cell_type

    @property
    def walls(self) -> Annotated[np.typing.NDArray[np.int_], (2,)]:
        return self._walls

    @property
    def holds_agent(self) -> bool:
        return self._holds_agent

    @property
    def holds_target(self) -> bool:
        return self._holds_target

    # setters
    @cell_type.setter
    def cell_type(self, cell_type: int) -> None:
        assert (
            isinstance(cell_type, int) and cell_type in cell_types
        ), f"cell_type must be an int in range {min(cell_types)}-{max(cell_types)}"
        self._cell_type = cell_type

    @walls.setter
    def walls(self, walls: Annotated[np.typing.NDArray[np.int_], (2,)]) -> None:
        assert (
            isinstance(walls, np.ndarray)
            and np.all(np.isin(walls, [-1, 0, 1]))
            and list(walls.shape) == [2]
        ), "ValueError: walls must be a numpy array with values in {-1, 0, 1} and shape (,2)"
        self._walls = walls

    @holds_agent.setter
    def holds_agent(self, holds_agent: bool) -> None:
        assert isinstance(holds_agent, bool), "holds_agent must be a bool"
        self._holds_agent = holds_agent

    @holds_target.setter
    def holds_target(self, holds_target: bool) -> None:
        assert isinstance(holds_target, bool), "holds_target must be a bool"
        self._holds_target = holds_target


# c = Cell((0, 0), 1, (0, 0, 0, 1))
# print(c.cell_type)
# c.cell_type = 2
# print(c.walls)
# print(c)
