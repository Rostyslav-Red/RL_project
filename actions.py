from enum import Enum


class Actions(Enum):
    LEFT = (0, -1)
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    # STAY = (0, 0)

    @staticmethod
    def get_by_index(index: int) -> 'Actions':
        actions = [Actions.LEFT, Actions.UP, Actions.RIGHT, Actions.DOWN]
        return actions[index]