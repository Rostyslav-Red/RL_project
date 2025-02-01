from typing import List, Tuple, Annotated, Optional, Any, SupportsFloat
from unittest import case
from gymnasium.core import ObsType, ActType, RenderFrame
from cell import Cell
from constants import *
from copy import deepcopy
from itertools import chain
import numpy as np
import random
import gymnasium as gym
from actions import Actions
from functools import reduce


class Board(gym.Env):
    def __init__(self, board: List[List[Cell]]):
        super(Board, self).__init__()

        # Make sure all rows have the same length
        assert len(reduce(lambda x, y: x if len(x) == len(y) else [], board)) != 0, "Rows don't have the same length."
        self._board_size: Tuple[int, int] = (len(board), len(board[0]))

        self._board: List[List[Cell]] = board
        self._cat_position: Optional[Annotated[np.typing.NDArray[np.int_], (2,)]] = None
        self._target_position: Optional[Annotated[np.typing.NDArray[np.int_], (2,)]] = (
            None
        )

        # Define Gymnasium Environment variables
        self.action_space = gym.spaces.Discrete(len(Actions))

        self.observation_space = gym.spaces.Dict({
            "agent_pos": gym.spaces.Box(0, max(self._board_size) - 1, shape=(2,), dtype=int),
            "target_pos": gym.spaces.Box(0, max(self._board_size) - 1, shape=(2,), dtype=int)
        })

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

    def __getitem__(self, item: Annotated[np.typing.NDArray[np.int_], (2,)]):
        return self._board[item[0]][item[1]]

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
    def possible_moves(self) -> List[Annotated[np.typing.NDArray[np.int_], (2,)]]:
        """
        Returns a list of directions in which the cat can move from a current position.
        :return:
        """
        possible_directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        possible_destinations = [
            self._cat_position + np.array(direction)
            for direction in possible_directions
        ]
        updated_destinations = [np.array([0, 0])]

        for direction, destination in zip(possible_directions, possible_destinations):
            ind = np.nonzero(np.array(direction))[0][0]

            if (
                0 <= destination[0] < self._board_size[0]
                and 0 <= destination[1] < self._board_size[1]
                and direction[ind]
                != self._board[self._cat_position[0]][self._cat_position[1]].walls[ind]
            ):
                updated_destinations.append(np.array(direction))
        return updated_destinations

    def move(self, direction: Annotated[np.typing.NDArray[np.int_], (2,)]):
        # moves the cat in the specified direction
        if not self._move(direction):
            return False

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
            [0, 0],
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
        if not move_target and not direction.tolist() == [0, 0]:
            ind = np.nonzero(direction)[0][0]
            if direction[ind] == current_cell.walls[ind]:
                return
        # ensures the mouse cannot go on the tree cell
        else:
            if self._board[possible_location[0]][possible_location[1]].cell_type == 1:
                return False

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

    # Gymnasium Support methods
    def _get_obs(self) -> ObsType:
        return {"agent_pos": self._cat_position,
                "target_pos": self._target_position}

    def _get_info(self) -> dict[str, Any]:
        return {"Info": 0}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        if seed:
            np.random.seed(seed)

        # Position is given
        if options and "cat_position" in options.keys() and "target_position" in options.keys():
            self._cat_position: Annotated[np.typing.NDArray[np.int_], (2,)] = options["cat_position"]
            self._target_position: Annotated[np.typing.NDArray[np.int_], (2,)] = options["target_position"]

        # Position not given, initialise randomly
        else:
            # Initialise cat position
            self._cat_position: Annotated[np.typing.NDArray[np.int_], (2,)] = np.array(
                [np.random.randint(0, self._board_size[0]),
                 np.random.randint(0, self._board_size[1])]
            )
            # Initialise target position
            self._target_position: Annotated[np.typing.NDArray[np.int_], (2,)] = np.array(
                [np.random.randint(0, self._board_size[0]),
                 np.random.randint(0, self._board_size[1])]
            )

        self[self._cat_position].holds_agent = True
        self[self._target_position].holds_target = True

        return self._get_obs(), self._get_info()

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        direction = np.array(Actions.get_by_index(action).value)
        self.move(direction)

        current_cell_type = cell_types[self[self._cat_position].cell_type]

        observation = self._get_obs()
        terminated = np.all(self._cat_position == self._target_position)
        reward = rewards["caught"] if terminated else rewards[current_cell_type]
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return str(self)


    # Factory method
    @staticmethod
    def board_factory(
            board: List[List[Cell]],
            *,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None
    ) -> Tuple['Board', ObsType, dict[str, Any]]:

        board = Board(board)
        obs, info = board.reset(seed=seed, options=options)
        return board, obs, info
