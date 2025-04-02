from agent import RandomAgent, HumanAgent, PolicyAgent
from deep_qlearning import DeepQLearningAgent, get_data_and_train, RLData
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from dynamic_programming_policy import DynamicProgrammingPolicy
from policy import Policy
from temporal_difference_policy import TemporalDifferencePolicy
from typing import Any
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt


def plot_rewards_over_time(env: gym.Env, policy: Policy, n_episodes: int, plot_save_path: str, algorithm_name: str,
                           max_steps: int = 10, options: dict[str, Any] = None):
    """
    Plots mean of the rewards obtained by a policy over time.

    @param env: The environment on which the policy should act.
    @param policy: The policy for which data is plotted.
    @param n_episodes: Number of episodes sampled.
    @param plot_save_path: Path where the plot is saved.
    @param algorithm_name: Name of the algorithm used to create the policy.
    @param max_steps: Maximum number of steps before truncating the episode.
    @param options: Environment options given to it in env.reset().
    """
    env = TimeLimit(env, max_episode_steps=max_steps)
    data = np.zeros((n_episodes, max_steps))
    for i in range(n_episodes):
        _, _, rewards, _, _ = PolicyAgent.sample_episode(env, policy, reset_options=options)
        rewards += tuple(0 for _ in range(max_steps - len(rewards)))
        data[i, :] = np.array(rewards)

    plt.figure()
    x = range(1, max_steps + 1)
    y = data.mean(axis=0)
    plt.bar(x, height=y)
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.title(f"Average reward of {algorithm_name} at each time step.")
    plt.savefig(plot_save_path)


if __name__ == "__main__":
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    options = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}

    board = gym.make("Board-v0", render_mode=None)

    # Baseline
    random_policy = Policy(board.observation_space, board.action_space)

    plot_rewards_over_time(board, random_policy, n_episodes=1000,
                           plot_save_path="plots/r_over_time/random_policy_r_over_time.png", options=options,
                           algorithm_name="Random Policy")

    # Dynamic Programming
    value_iteration = DynamicProgrammingPolicy(
        board, algorithm="ValueIteration", discount=0.9, stopping_criterion=0.00000001
    )

    plot_rewards_over_time(board, value_iteration, n_episodes=1000,
                           plot_save_path="plots/r_over_time/value_iteration_r_over_time.png", options=options,
                           algorithm_name="Value Iteration")

    policy_iteration = DynamicProgrammingPolicy(
        board, algorithm="PolicyIteration", discount=0.9, stopping_criterion=0.00000001
    )

    plot_rewards_over_time(board, policy_iteration, n_episodes=1000,
                           plot_save_path="plots/r_over_time/policy_iteration_r_over_time.png", options=options,
                           algorithm_name="Policy Iteration")

    # Monte Carlo goes here

    # Temporal Difference Learning
    sarsa = TemporalDifferencePolicy(board.observation_space, board.action_space, algorithm="SARSA", env=board,
                                     n_episodes=1000, alpha=0.5, gamma=0.9)

    plot_rewards_over_time(board, sarsa, n_episodes=1000,
                           plot_save_path="plots/r_over_time/sarsa_r_over_time.png", options=options,
                           algorithm_name="SARSA")

    q_learning = TemporalDifferencePolicy(board.observation_space, board.action_space, algorithm="QLearning", env=board,
                                          n_episodes=1000, alpha=0.5, gamma=0.9)

    plot_rewards_over_time(board, sarsa, n_episodes=1000,
                           plot_save_path="plots/r_over_time/q_learning_r_over_time.png", options=options,
                           algorithm_name="Q-Learning")

    # Deep Q Learning, note the model is not trained here, so this file does not take centuries to run.
    agent = DeepQLearningAgent.load(board, "policies/model.pt")

    # Uncomment the line below to train a new model from scratch, these are the setting used
    # agent = get_data_and_train(board, (10, 10), n_episodes=100000, batch_size=1024)

    deep_q_policy = agent.make_greedy_tabular_policy()

    plot_rewards_over_time(board, deep_q_policy, n_episodes=1000,
                           plot_save_path="plots/r_over_time/deep_q_learning_r_over_time.png", options=options,
                           algorithm_name="Deep Q-Learning")

