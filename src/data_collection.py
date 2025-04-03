from agent import PolicyAgent
from deep_qlearning import DeepQLearningAgent, get_data_and_train, RLData
import numpy as np
import gymnasium as gym
from dynamic_programming_policy import DynamicProgrammingPolicy
from policy import Policy
from src.monte_carlo_policy import MonteCarloPolicy
from temporal_difference_policy import TemporalDifferencePolicy
from typing import Any, Optional, Callable
from gymnasium.wrappers import TimeLimit
import matplotlib.pyplot as plt
from threading import Thread


def plot_rewards_over_time(env: gym.Env, policy: Policy, n_episodes: int, plot_save_path: str, algorithm_name: str,
                           max_steps: int = 10, options: dict[str, Any] = None, seed: Optional[int] = None):
    """
    Plots mean of the rewards obtained by a policy over time.

    @param env: The environment on which the policy should act.
    @param policy: The policy for which data is plotted.
    @param n_episodes: Number of episodes sampled.
    @param plot_save_path: Path where the plot is saved.
    @param algorithm_name: Name of the algorithm used to create the policy.
    @param max_steps: Maximum number of steps before truncating the episode.
    @param seed: Random seed, optional.
    @param options: Environment options given to it in env.reset().
    """
    rng = np.random.default_rng(seed)

    env = TimeLimit(env, max_episode_steps=max_steps)
    data = np.zeros((n_episodes, max_steps))
    for i in range(n_episodes):
        _, _, rewards, _, _ = PolicyAgent.sample_episode(env, policy, reset_options=options, seed=int(rng.integers(low=0, high=1000000)))
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


def collect_algorithm_data(
        data: np.array,
        env: gym.Env,
        ticks: tuple[int],
        n_samples: int,
        policy_train_function: Callable[[int], Policy],
        algorithm_name: str,
        options: dict[str, Any] = None,
        seed: Optional[int] = None):

    for i, tick in enumerate(ticks):
        print(f"Algorithm {algorithm_name} at tick {tick}")
        rng = np.random.default_rng(seed)

        eval_policy = policy_train_function(tick)

        for j in range(n_samples):
            _, _, rewards, _, _ = PolicyAgent.sample_episode(env, eval_policy, reset_options=options,
                                                             seed=int(rng.integers(low=0, high=1000000)))
            data[i, j] = sum(rewards)

    return data.mean(axis=0)


def plot_reward_evolution_comparison(
        env: gym.Env,
        ticks: tuple[int],
        n_samples: int,
        policy_train_functions: tuple[Callable[[int], Policy]],
        algorithm_names: tuple[str, ...],
        plot_save_path: str,
        max_steps: Optional[int] = None,
        plot_title: str = "Episode Reward Over Time",
        options: dict[str, Any] = None,
        seed: Optional[int] = None):

    data = ()
    threads = []
    if max_steps:
        env = TimeLimit(env, max_episode_steps=max_steps)

    for func, algorithm_name in zip(policy_train_functions, algorithm_names):
        algorithm_data = np.zeros((len(ticks), n_samples))
        thread = Thread(target=collect_algorithm_data, args=(algorithm_data, env, ticks, n_samples, func, algorithm_name, options, seed))
        thread.start()
        threads.append(thread)
        data += ((algorithm_data, algorithm_name),)

    for thread in threads:
        thread.join()

    print("\nFinished training and collecting, making final plot.")
    plt.figure(figsize=(20, 10))

    for algorithm_data, algorithm in data:
        plt.plot(ticks, algorithm_data.mean(axis=1), label=algorithm, linewidth=2)

    plt.title(plot_title, fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.savefig(plot_save_path)

if __name__ == "__main__":
    gym.register(id="Board-v0", entry_point="board:ConfiguredBoard")

    options = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}

    board = gym.make("Board-v0", render_mode="None")

    print("Creating Plots")
    print("Creating Reward over Time plots (0/7)")
    # Baseline
    random_policy = Policy(board.observation_space, board.action_space)

    plot_rewards_over_time(board, random_policy, n_episodes=1000,
                           plot_save_path="../plots/r_over_time/random_policy_r_over_time.png", options=options,
                           algorithm_name="Random Policy", seed=0)
    print("Finished plotting random (1/7)")

    # Dynamic Programming
    value_iteration = DynamicProgrammingPolicy(
        board, algorithm="ValueIteration", discount=0.9, stopping_criterion=0.00000001
    )

    plot_rewards_over_time(board, value_iteration, n_episodes=1000,
                           plot_save_path="../plots/r_over_time/value_iteration_r_over_time.png", options=options,
                           algorithm_name="Value Iteration", seed=0)
    print("Finished plotting value iteration (2/7)")

    policy_iteration = DynamicProgrammingPolicy(
        board, algorithm="PolicyIteration", discount=0.9, stopping_criterion=0.00000001
    )

    plot_rewards_over_time(board, policy_iteration, n_episodes=1000,
                           plot_save_path="../plots/r_over_time/policy_iteration_r_over_time.png", options=options,
                           algorithm_name="Policy Iteration", seed=0)

    print("Finished plotting policy iteration (3/7)")

    # Monte Carlo goes here
    monte_carlo = MonteCarloPolicy(
        board.observation_space, board.action_space, algorithm="FirstVisitEpsilonGreedy", env=board, n_episodes=100,
        gamma=0.9, epsilon=0.3
       )

    plot_rewards_over_time(board, monte_carlo, n_episodes=1000,
                           plot_save_path="../plots/r_over_time/monte_carlo_r_over_time.png", options=options,
                           algorithm_name="First Visit Epsilon Greedy Monte Carlo", seed=0)

    print("Finished plotting Monte Carlo (4/7)")

    # Temporal Difference Learning
    sarsa = TemporalDifferencePolicy(board.observation_space, board.action_space, algorithm="SARSA", env=board,
                                     n_episodes=100, alpha=0.5, gamma=0.9)

    plot_rewards_over_time(board, sarsa, n_episodes=1000,
                           plot_save_path="../plots/r_over_time/sarsa_r_over_time.png", options=options,
                           algorithm_name="SARSA", seed=0)

    print("Finished plotting SARSA (5/7)")

    q_learning = TemporalDifferencePolicy(board.observation_space, board.action_space, algorithm="QLearning", env=board,
                                          n_episodes=100, alpha=0.5, gamma=0.9)

    plot_rewards_over_time(board, q_learning, n_episodes=1000,
                           plot_save_path="../plots/r_over_time/q_learning_r_over_time.png", options=options,
                           algorithm_name="Q-Learning", seed=0)

    print("Finished plotting Q-Learning (6/7)")

    # Deep Q Learning, note the model is not trained here, so this file does not take centuries to run.
    agent = DeepQLearningAgent.load(board, "../policies/model.pt")

    # Uncomment the line below to train a new model from scratch, these are the setting used
    # agent = get_data_and_train(board, (10, 10), n_episodes=100000, batch_size=1024)

    deep_q_policy = agent.make_greedy_tabular_policy()

    plot_rewards_over_time(board, deep_q_policy, n_episodes=1000,
                           plot_save_path="../plots/r_over_time/deep_q_learning_r_over_time.png", options=options,
                           algorithm_name="Deep Q-Learning", seed=0)

    print("Finished plotting Deep Q-Learning (7/7)")
    print("_" * 100, "\n")
    print("Making comparison plot of MC, SARSA, and Q-Learning")
    # Comparison plots
    mc_comparison_policy = MonteCarloPolicy(board.observation_space, board.action_space)
    q_learning_comparison_policy = TemporalDifferencePolicy(board.observation_space, board.action_space)
    sarsa_comparison_policy = TemporalDifferencePolicy(board.observation_space, board.action_space)

    policy_funcs = (
        lambda n_episodes: mc_comparison_policy.first_visit_monte_carlo_control(board, n_episodes=n_episodes, gamma=0.9,
                                                                                reset=False, epsilon=0.3),
        lambda n_episodes: q_learning_comparison_policy.q_learning(board, n_episodes=n_episodes, alpha=0.5, gamma=0.9, reset=False),
        lambda n_episodes: sarsa_comparison_policy.sarsa(board, n_episodes=n_episodes, alpha=0.5, gamma=0.9,
                                                         reset=False)
    )
    algorithms = ("Monte-Carlo", "Q-Learning", "SARSA")

    plot_reward_evolution_comparison(board, ticks=tuple(range(0, 100, 1)), n_samples=10,
                                     policy_train_functions=policy_funcs, algorithm_names=algorithms,
                                     plot_save_path="../plots/comparison/comparison_plot_final.png", options=options, seed=0)

    print("Finished")