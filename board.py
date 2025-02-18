from typing import List, Tuple, Annotated, Optional, Any, SupportsFloat
from gymnasium.core import ObsType, ActType, RenderFrame
from cell import Cell
from constants import *
import numpy as np
import gymnasium as gym
from actions import Actions
from functools import reduce
import warnings


class Board(gym.Env):
    """
    Abstract Gymnasium environment for a board with a cat and a mouse. Cannot be registered using gymnasium.register().
    Inherit from this class and create a board in its constructor, making sure the child's constructor does not take
    arguments. These children can then be registered.
    """

    metadata = {"render_modes": ("human",)}

    def __init__(self, board: List[List[Cell]], render_mode: str):
        super(Board, self).__init__()

        self.render_mode = render_mode

        # Make sure all rows have the same length
        assert (
            len(reduce(lambda x, y: x if len(x) == len(y) else [], board)) != 0
        ), "Rows don't have the same length."
        self._board_size: Annotated[np.typing.NDArray[np.int_], (2,)] = np.array(
            [len(board), len(board[0])]
        )

        self._board: List[List[Cell]] = board
        self._cat_position: Optional[Annotated[np.typing.NDArray[np.int_], (2,)]] = None
        self._target_position: Optional[Annotated[np.typing.NDArray[np.int_], (2,)]] = (
            None
        )

        # Define Gymnasium Environment variables
        self.action_space = gym.spaces.Discrete(len(Actions))

        self.observation_space = gym.spaces.Dict(
            {
                "agent_pos": gym.spaces.Box(
                    0, np.max(self._board_size) - 1, shape=(2,), dtype=int
                ),
                "target_pos": gym.spaces.Box(
                    0, np.max(self._board_size) - 1, shape=(2,), dtype=int
                ),
            }
        )

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

        return "\n" + result

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
    def possible_mouse_destinations(
        self, current_position
    ) -> List[Annotated[np.typing.NDArray[np.int_], (2,)]]:
        """
        Returns a list of destinations in which the mouse can move from a current position.
        :return:
        """
        possible_directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        possible_destinations = [
            current_position + np.array(direction) for direction in possible_directions
        ]
        updated_destinations = []

        for direction, destination in zip(possible_directions, possible_destinations):

            if (
                0 <= destination[0] < self._board_size[0]
                and 0 <= destination[1] < self._board_size[1]
                and self._board[destination[0]][destination[1]].cell_type != 1
            ):
                updated_destinations.append(np.array(destination))
        return updated_destinations

    def possible_cat_destinations(
        self, current_position
    ) -> List[Annotated[np.typing.NDArray[np.int_], (2,)]]:
        """
        Returns a list of directions in which the cat can move from a current position.
        :return:
        """
        possible_directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        possible_destinations = [
            current_position + np.array(direction) for direction in possible_directions
        ]
        updated_directions = [np.array([0, 0])]

        for direction, destination in zip(possible_directions, possible_destinations):
            ind = np.nonzero(np.array(direction))[0][0]

            if (
                0 <= destination[0] < self._board_size[0]
                and 0 <= destination[1] < self._board_size[1]
                and direction[ind]
                != self._board[current_position[0]][current_position[1]].walls[ind]
            ):
                updated_directions.append(np.array(destination))
        return updated_directions

    def move(self, direction: Annotated[np.typing.NDArray[np.int_], (2,)]) -> bool:
        # moves the cat in the specified direction
        moved = self._move(direction)

        # moves the mouse in a random allowed direction. If none are allowed, the mouse doesn't move
        possible_directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        self.np_random.shuffle(possible_directions)

        for target_direction in possible_directions:
            if self._move(np.array(target_direction), move_target=True):
                return moved
        return moved

    def _move(
        self,
        direction: Annotated[np.typing.NDArray[np.int_], (2,)],
        move_target: bool = False,
    ) -> bool:
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
        if not (
            (
                np.zeros(
                    2,
                )
                <= possible_location
            ).all()
            and (possible_location < self._board_size).all()
        ):
            return False

        current_cell = self._board[location[0]][location[1]]

        # ensures the cat cannot move through the walls withing the level
        if not move_target and not direction.tolist() == [0, 0]:
            ind = np.nonzero(direction)[0][0]
            if direction[ind] == current_cell.walls[ind]:
                return False
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
        return {"agent_pos": self._cat_position, "target_pos": self._target_position}

    def _get_info(self) -> dict[str, Any]:
        return {"Info": 0}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[ObsType, dict[str, Any]]:

        super().reset(seed=seed)

        # Remove cat and mouse from board
        if self._cat_position is not None:
            self[self._cat_position].holds_agent = False
        if self._target_position is not None:
            self[self._target_position].holds_target = False

        position_reset = False
        # Position is given
        if (
            options
            and "cat_position" in options.keys()
            and "target_position" in options.keys()
        ):
            # Position is valid
            if isinstance(options["cat_position"], np.ndarray) and isinstance(
                options["target_position"], np.ndarray
            ):
                self._cat_position: Annotated[np.typing.NDArray[np.int_], (2,)] = (
                    options["cat_position"]
                )
                self._target_position: Annotated[np.typing.NDArray[np.int_], (2,)] = (
                    options["target_position"]
                )
                position_reset = True
            # Position is given but invalid, randomise instead.
            else:
                warnings.warn(
                    "Options dictionary provided but positions are not np.ndarray. Randomising positions instead."
                )

        # Position not given/invalid, initialise randomly
        if not position_reset:
            # Initialise cat position
            self._cat_position: Annotated[np.typing.NDArray[np.int_], (2,)] = np.array(
                [
                    self.np_random.integers(0, self._board_size[0]),
                    self.np_random.integers(0, self._board_size[1]),
                ]
            )
            # Initialise target position
            self._target_position: Annotated[np.typing.NDArray[np.int_], (2,)] = (
                np.array(
                    [
                        self.np_random.integers(0, self._board_size[0]),
                        self.np_random.integers(0, self._board_size[1]),
                    ]
                )
            )

        self[self._cat_position].holds_agent = True
        self[self._target_position].holds_target = True

        return self._get_obs(), self._get_info()

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Move agent in given direction.
        direction = np.array(Actions.get_by_index(action).value)
        self.move(direction)

        current_cell_type = cell_types[self[self._cat_position].cell_type]

        # Create return variables
        observation = self._get_obs()
        terminated = np.all(self._cat_position == self._target_position)
        reward = rewards["caught"] if terminated else rewards[current_cell_type]
        truncated = False
        info = self._get_info()

        # Print board if rendering is set to human.
        if self.render_mode == "human":
            print(self)

        return observation, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return str(self)


class ConfiguredBoard(Board):

    def __init__(self, render_mode: str = "human"):
        # An empty board of the board_size
        example_board = [
            [Cell(position=(j, i)) for i in range(board_size[1])]
            for j in range(board_size[0])
        ]

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

        super().__init__(example_board, render_mode)
