from typing import List, Tuple, Annotated, Optional, Any, SupportsFloat, Dict, Union
from gymnasium.core import ObsType, ActType, RenderFrame

import actions
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

    metadata = {"render_modes": ("human", "simple")}

    def __init__(self, board: List[List[Cell]], render_mode: str):
        super(Board, self).__init__()

        # Ensures that the render mode is appropriate
        assert (
            render_mode in Board.metadata["render_modes"]
        ), f"ValueError: {render_mode} is not a valid render_mode"
        self.render_mode = render_mode

        # Ensures all rows have the same length (it also ensures the same thing for the columns)
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
                    0, np.max(self._board_size) - 1, shape=(2,), dtype=np.int_
                ),
                "target_pos": gym.spaces.Box(
                    0, np.max(self._board_size) - 1, shape=(2,), dtype=np.int_
                ),
            }
        )

        # the dynamics of the environment. Read find_dynamics() docs for more info
        self.P: Dict[
            Tuple[Tuple[int, int, int, int], int], Tuple[Tuple[int, int, int, int], ...]
        ] = self.find_dynamics()

        # the reward space of the environment. Read find_reward_space() docs for more info
        self.R: Dict[Tuple[int, int, int, int], float] = self.find_reward_space()

        self.rng = np.random.default_rng()

    # dunder methods
    def __str__(self):
        return self.render()

    def __getitem__(
        self, item: Union[Annotated[np.typing.NDArray[np.int_], (2,)], Tuple[int, int]]
    ):
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

                if (self._target_position == self._cat_position).all():
                    warnings.warn(
                        "Cat and mouse position initialised to same position, randomising instead."
                    )
                else:
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

            # Make sure board doesn't randomise to a terminal state.
            if (self._target_position == self._cat_position).all():
                return self.reset(seed=seed, options=options)

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
        print(self.render())

        return observation, reward, terminated, truncated, info

    # Rendering
    def _human_render(self):
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

    def _simple_render(self):
        """
        represents the board as a matrix of integers, where:
            - 0 corresponds to an empty cell
            - 1 corresponds to a tree
            - 2 corresponds to an empty cell with an agent
            - 3 corresponds to a tree with an agent
            - 4 corresponds to an empty cell with a target
            - 5 corresponds to a tree with a target
            - 6 corresponds to an empty cell with both the agent and the target
            - 7 corresponds to a tree with both the agent and the target
            - 8 corresponds to an in-between space with a wall
            - 9 corresponds to an in-between space without a wall
        :return: a string representation of the board
        """
        result = ""
        for row in self._board:
            walls = "9"
            if row[0].walls[1] == -1:
                cells = "8"
            else:
                cells = "9"
            for cell in row:
                if cell.walls[0] == -1:
                    walls += "89"
                else:
                    walls += "99"

                match (cell.cell_type, cell.holds_agent, cell.holds_target):
                    case (0, 0, 0):
                        cells += "0"
                    case (1, 0, 0):
                        cells += "1"
                    case (0, 1, 0):
                        cells += "2"
                    case (1, 1, 0):
                        cells += "3"
                    case (0, 0, 1):
                        cells += "4"
                    case (1, 0, 1):
                        cells += "5"
                    case (0, 1, 1):
                        cells += "6"
                    case (1, 1, 1):
                        cells += "7"

                if cell.walls[1] == 1:
                    cells += "8"
                else:
                    cells += "9"
            result += walls + "\n" + cells + "\n"

        result += "9"
        final_row = self._board[-1]
        for cell in final_row:
            if cell.walls[0] == 1:
                result += "89"
            else:
                result += "99"
        return result

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "human":
            return self._human_render()
        elif self.render_mode == "simple":
            return self._simple_render()
        else:
            raise Exception("Unknown render mode")

    # Other
    def find_dynamics(
        self,
    ) -> Dict[
        Tuple[Tuple[int, int, int, int], int], Tuple[Tuple[int, int, int, int], ...]
    ]:
        """
        Returns a variable, representing the dynamics of the environment in the following format:
        {((0, 1, 2, 3), 2) : ((0, 1, 2, 3), (0, 1, 2, 3), ...)}
        where the key is a tuple of a current state of the board* and an integer representing the action (0-3).
        The board state is a tuple, consisting of the coordinates of the cat (first two integers), and the
        coordinates of the target (last two integers).
        The values of the dictionary are a tuple of tuples, where each inner tuple represents a board state, which
        can occur from taking an action (from the key) in a state (in the key)
        """
        all_states = []
        result = {}

        # create the tuples, representing all positions (4x4 x 4x4)
        for x1 in range(0, self._board_size[0]):
            for y1 in range(0, self._board_size[1]):
                for x2 in range(0, self._board_size[0]):
                    for y2 in range(0, self._board_size[1]):
                        all_states.append((x1, y1, x2, y2))

        # define the key structure of the final dictionary
        for state in all_states:
            for direction in range(4):
                result[(state, direction)] = ()

        # defining the values of the final dictionary
        for state, action in result:
            # if we're in a terminal state, assign an empty tuple as a value
            if state[:2] == state[2:]:
                result[(state, action)] = tuple()
                continue

            # a flag determining whether a cat actually moves when taking this action (or if it bumps into the wall)
            valid_cat_move = True
            # the coordinates of the cat after taking a certain action
            landing_x = state[0] + directions[action][0]
            landing_y = state[1] + directions[action][1]

            # if given action moves the cat beyond the bounds of the board, the flag is set to False
            if landing_x < 0 or landing_x >= self._board_size[0]:
                valid_cat_move = False
            if landing_y < 0 or landing_y >= self._board_size[1]:
                valid_cat_move = False

            # if the cat is trying to move into a wall, the flag is set to False
            cat_cell = self[state[0], state[1]]

            if (
                directions[action][0] != 0
                and cat_cell.walls[0] == directions[action][0]
                or directions[action][1] != 0
                and cat_cell.walls[1] == directions[action][1]
            ):
                valid_cat_move = False

            # define the coordinates of the cat position after taking a move
            if valid_cat_move:
                cat_pos = (landing_x, landing_y)
            else:
                cat_pos = (state[0], state[1])

            # find all the possible target positions following the given target position
            possible_target_locations = []
            for target_action in range(4):
                # the coordinates of the target after 'taking' a certain action
                target_landing_x = state[2] + directions[target_action][0]
                target_landing_y = state[3] + directions[target_action][1]

                # if the action results in the target moving beyond the scope of the board, move onto the next action
                if target_landing_x < 0 or target_landing_x >= self._board_size[0]:
                    continue
                if target_landing_y < 0 or target_landing_y >= self._board_size[1]:
                    continue
                # if the action results in the target being in a tree, move onto the next action
                if self[target_landing_x, target_landing_y].cell_type == 1:
                    continue

                # otherwise, add the resulting target location to the list of possible locations
                possible_target_locations.append((target_landing_x, target_landing_y))

            # if the mouse can't move at all, its only possible position is its current position
            if not possible_target_locations:
                possible_target_locations.append((state[2], state[3]))

            # determine the value of the dictionary
            value = tuple(
                cat_pos + target_pos for target_pos in possible_target_locations
            )

            # set this value to the appropriate key in the final dictionary
            result[(state, action)] = value
        return result

    def find_reward_space(self) -> Dict[Tuple[int, int, int, int], float]:
        """
        Finds a reward space of the environment in the following format:
        {(1, 2, 3, 4): 5}
        where the key is a tuple representing a current state of the board*.
        The board state is a tuple, consisting of the coordinates of the cat (first two integers), and the
        coordinates of the target (last two integers).
        The values of the dictionary are the rewards, associated with the board state from key.
        :return:
        """
        result = {}
        all_states = set()

        # extracting all unique states from the P of the environment
        for state, _ in self.P:
            all_states.add(state)
        all_states = tuple(all_states)

        # assigning a reward to each state
        for state in all_states:
            if state[:2] == state[2:]:
                reward = rewards["caught"]
                result[state] = reward
                continue

            if self[state[0], state[1]].cell_type == 0:
                reward = rewards["empty"]
                result[state] = reward
                continue

            if self[state[0], state[1]].cell_type == 1:
                reward = rewards["tree"]
                result[state] = reward
                continue

        return result


class ConfiguredBoard(Board):
    """
    Defines a concrete board configuration that is used for testing throughout the project
    """

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
