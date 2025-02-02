import numpy as np

cell_types = {0: "empty", 1: "tree"}
cell_representations = {0: "·", 1: "○"}
rewards = {"caught": 20, "tree": -0.1, "empty": -1}
board_size = (4, 4)
