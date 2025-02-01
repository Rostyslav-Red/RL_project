from enum import Enum


class Actions(Enum):
    LEFT = (-1, 0)
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    STAY = (0, 0)

    @staticmethod
    def get_by_index(index: int) -> 'Actions':
        actions = [Actions.LEFT, Actions.UP, Actions.RIGHT, Actions.DOWN, Actions.STAY]
        return actions[index]