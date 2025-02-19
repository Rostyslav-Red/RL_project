import numpy as np

cell_types = {0: "empty", 1: "tree"}
cell_representations = {0: "·", 1: "○"}
rewards = {"caught": 20, "tree": -0.1, "empty": -1}
board_size = (4, 4)
directions = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (1, 0)}

"""
            - (0, 1) -> right
            - (0, -1) -> left
            - (1, 0) -> down
            - (-1, 0) -> up

            case "right":
                return 2
            case "left":
                return 0
            case "up":
                return 1
            case "down":
                return 3

"""
