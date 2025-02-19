from agent import RandomAgent, HumanAgent, PolicyAgent
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from policy import Policy

if __name__ == "__main__":
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    options = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}

    board = gym.make("Board-v0")

    # print(check_env(board.unwrapped))

    # !!! the following block of code is only for demonstrating __policy_evaluation(). Remove after viewing
    board.reset(
        options={"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}
    )

    p = Policy(board, seed=0)
    print(p.items())
    print(p._Policy__policy_evaluation())
    # !!! the of the block

    obs, _ = board.reset(seed=1)

    # Possible agents: HumanAgent, RandomAgent, PolicyAgent
    agent = PolicyAgent(board, Policy(board, seed=1))
    # agent = HumanAgent(board)

    print(f"Obtained reward: {agent.run_agent(obs)}")
