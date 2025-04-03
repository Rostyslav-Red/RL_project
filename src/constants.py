import numpy as np
from enum import Enum

cell_types = {0: "empty", 1: "tree"}
cell_representations = {0: "·", 1: "○"}
rewards = {"caught": 20, "tree": -0.1, "empty": -1}
board_size = (4, 4)
directions = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (1, 0)}

class Actions(Enum):
    LEFT = (0, -1)
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    # STAY = (0, 0)

    @staticmethod
    def get_by_index(index: int) -> "Actions":
        actions = [Actions.LEFT, Actions.UP, Actions.RIGHT, Actions.DOWN]
        return actions[index]
