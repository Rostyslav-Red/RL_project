from agent import RandomAgent, HumanAgent, PolicyAgent
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from policy import Policy

if __name__ == "__main__":
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    options = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}

    board = gym.make("Board-v0")

    # Checks if board is a valid environment
    # print(check_env(board.unwrapped))

    ### Block for computing policy, only run when computing a new policy
    p = Policy(board).value_iteration()
    p.save("improved_policy.json")
    ###

    obs, _ = board.reset()

    p = Policy.load(board, "improved_policy.json")
    # Possible agents: HumanAgent, RandomAgent, PolicyAgent
    agent = PolicyAgent(board, p)

    print(f"Obtained reward: {agent.run_agent(obs)}")
