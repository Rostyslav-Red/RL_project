import re
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


def collect_algorithm_data(env: gym.Env, policy: Policy, n_episodes: int, max_steps: int = 1000,
                           options: dict[str, Any] = None, seed: Optional[int] = None) -> np.ndarray:
    """
    Collects data for the given algorithm.

    @param env: Environment the data will be collected from.
    @param policy: Policy used for taking actions in the environment.
    @param n_episodes: Number of episodes to sample.
    @param max_steps: Maximum number of steps taken before truncation.
    @param options: Environment reset options.
    @param seed: Random seed.
    @return: A numpy array of shape (n_episodes, max_steps)
    """
    rng = np.random.default_rng(seed)

    env = TimeLimit(env, max_episode_steps=max_steps)
    data = np.zeros((n_episodes, max_steps))
    for i in range(n_episodes):
        _, _, rewards, _, _ = PolicyAgent.sample_episode(env, policy, reset_options=options,
                                                         seed=int(rng.integers(low=0, high=1000000)))
        rewards += tuple(0 for _ in range(max_steps - len(rewards)))
        data[i, :] = np.array(rewards)

    return data


def plot_rewards_over_time(data: dict[str, np.ndarray], plot_save_dir: str, max_steps: int = 10):
    """
    Plots mean of the rewards obtained by a policy over time.

    @param data: Data that is plotted. A dictionary mapping an algorithm to it's data.
    @param plot_save_dir: Directory where the plots are saved.
    @param max_steps: Maximum number of steps before truncating the episode.
    """
    for algorithm_name, algorithm_data in data.items():
        plt.figure(figsize=(10, 5))
        x = range(1, max_steps + 1)
        y = algorithm_data[:, :max_steps].mean(axis=0)
        plt.bar(x, height=y)
        plt.xlabel("Time")
        plt.ylabel("Reward")
        plt.title(f"Average reward of {algorithm_name} at each time step.")
        snake_case_algorithm_name = re.sub(r'(\s|-)', '_', algorithm_name).lower()
        plt.savefig(plot_save_dir + "/" + snake_case_algorithm_name + "_r_over_time.png")


def plot_comparison_barchart(data: dict[str, np.ndarray], plot_save_path: str) -> None:
    """
    Plots a barchart that can be used to compare returns of different RL algorithms.

    @param data: Data plotted, a dictionary mapping each algorithm to it's dataset.
    @param plot_save_path: Path where the plot is saved.
    """
    plt.figure(figsize=(10, 5))

    for algorithm, data in data.items():
        plt.bar(x=algorithm, height=data.sum(axis=1).mean())

    plt.xlabel("Algorithm")
    plt.ylabel("Average reward obtained")
    plt.title(f"Average reward of each algorithm")
    plt.savefig(plot_save_path)
    
    
def plot_reward_density(data: dict[str, np.ndarray], plot_save_dir):
    """
    Plots a histogram of returns.

    @param data: Data plotted from, a dictionary mapping each algorithm to its dataset.
    @param plot_save_dir: Directory where the plots will be saved.
    """
    for algorithm_name, algorithm_data in data.items():
        plt.figure(figsize=(10, 5))
        x = algorithm_data.sum(axis=1)
        plt.hist(x, bins=30, density=True)
        plt.xlabel("Reward")
        plt.ylabel("Probability")
        plt.title(f"Probability Density Function of {algorithm_name} Returns")
        snake_case_algorithm_name = re.sub(r'(\s|-)', '_', algorithm_name).lower()
        plt.savefig(plot_save_dir + "/" + snake_case_algorithm_name + "_return_probability.png")


def collect_algorithm_train_data(
        data: np.array,
        env: gym.Env,
        ticks: tuple[int],
        n_samples: int,
        policy_train_function: Callable[[int], Policy],
        algorithm_name: str,
        options: dict[str, Any] = None,
        seed: Optional[int] = None) -> np.ndarray:
    """
    Gradually trains an algorithm, collecting performance data every tick.

    @param data: np.array of data where the data will be stored. Has shape (len(ticks), n_samples).
    @param env: Environment the policy describes.
    @param ticks: Episode steps at which the algorithm will be evaluated.
    @param n_samples: Number of samples taken at each tick.
    @param policy_train_function: Function that trains a policy by tick episodes.
    @param algorithm_name: Name of the algorithm trained.
    @param options: Environment reset options.
    @param seed: Random seed.
    @return: Collected data, a numpy array of shape (len(ticks), n_samples)
    """

    print(f"Started training {algorithm_name}. This may take a while...")
    for i, tick in enumerate(ticks):
        rng = np.random.default_rng(seed)

        eval_policy = policy_train_function(tick)

        for j in range(n_samples):
            _, _, rewards, _, _ = PolicyAgent.sample_episode(env, eval_policy, reset_options=options,
                                                             seed=int(rng.integers(low=0, high=1000000)))
            data[i, j] = sum(rewards)

    print(f"Finished training {algorithm_name}.")
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
    """
    Plots the reward evolution of a set of algorithms.

    @param env: Environment the policy describes.
    @param ticks: Episode steps at which the algorithm will be evaluated.
    @param n_samples: Number of samples taken at each tick.
    @param policy_train_functions: Tuple of functions that train a given algorithm by tick steps.
    @param algorithm_names: Names of the algorithms trained.
    @param plot_save_path: Path where the plot will be saved.
    @param options: Environment reset options.
    @param seed: Random seed.
    @param max_steps: Maximum steps taken before truncation.
    @param plot_title: Title of the plot.
    """

    data = ()
    threads = []
    if max_steps:
        env = TimeLimit(env, max_episode_steps=max_steps)

    for func, algorithm_name in zip(policy_train_functions, algorithm_names):
        algorithm_data = np.zeros((len(ticks), n_samples))
        thread = Thread(target=collect_algorithm_train_data, args=(algorithm_data, env, ticks, n_samples, func, algorithm_name, options, seed))
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
    
    # Global parameters
    OPTIONS = {"cat_position": np.array([0, 0]), "target_position": np.array([3, 3])}
    BOARD = gym.make("Board-v0", render_mode="None")
    SEED = 0
    N_EPISODES = 1000
    MAX_STEPS = 100
    N_COMPARISON_SAMPLES = 10
    COMPARISON_TICKS = tuple(range(0, 101, 1))

    # Data collection
    print("Collecting data")
    data = {}

    random_policy = Policy(BOARD.observation_space, BOARD.action_space, seed=SEED)

    data["Random Policy"] = collect_algorithm_data(BOARD, random_policy, n_episodes=N_EPISODES, options=OPTIONS, seed=SEED, max_steps=MAX_STEPS)
    print("Finished collecting data for random (1/7)")

    # Dynamic Programming
    value_iteration = DynamicProgrammingPolicy(
        BOARD, algorithm="ValueIteration", discount=0.9, stopping_criterion=0.00000001, seed=SEED
    )

    data["Value Iteration"] = collect_algorithm_data(BOARD, value_iteration, n_episodes=N_EPISODES, options=OPTIONS, seed=SEED, max_steps=MAX_STEPS)
    print("Finished collecting data for value iteration (2/7)")

    policy_iteration_reward = policy_iteration = DynamicProgrammingPolicy(
        BOARD, algorithm="PolicyIteration", discount=0.9, stopping_criterion=0.00000001, seed=SEED
    )

    data["Policy Iteration"] = collect_algorithm_data(BOARD, policy_iteration, n_episodes=N_EPISODES, options=OPTIONS, seed=SEED, max_steps=MAX_STEPS)
    print("Finished collecting data for policy iteration (3/7)")

    # Monte Carlo goes here
    monte_carlo = MonteCarloPolicy(
        BOARD.observation_space, BOARD.action_space, algorithm="FirstVisitEpsilonGreedy", env=BOARD, n_episodes=100,
        gamma=0.9, epsilon=0.3, seed=SEED
    )

    data["Monte Carlo"] = collect_algorithm_data(BOARD, monte_carlo, n_episodes=N_EPISODES, options=OPTIONS, seed=SEED, max_steps=MAX_STEPS)
    print("Finished collecting data for Monte Carlo (4/7)")

    # Temporal Difference Learning
    sarsa = TemporalDifferencePolicy(BOARD.observation_space, BOARD.action_space, algorithm="SARSA", env=BOARD,
                                     n_episodes=100, alpha=0.5, gamma=0.9, epsilon=0.3, seed=SEED)

    data["SARSA"] = collect_algorithm_data(BOARD, sarsa, n_episodes=N_EPISODES, options=OPTIONS, seed=SEED, max_steps=MAX_STEPS)
    print("Finished collecting data for SARSA (5/7)")

    q_learning = TemporalDifferencePolicy(BOARD.observation_space, BOARD.action_space, algorithm="QLearning", env=BOARD,
                                          n_episodes=100, alpha=0.5, gamma=0.9, epsilon=0.3, seed=SEED)

    data["Q-Learning"] = collect_algorithm_data(BOARD, q_learning, n_episodes=N_EPISODES, options=OPTIONS, seed=SEED, max_steps=MAX_STEPS)
    print("Finished collecting data for Q-Learning (6/7)")

    # Deep Q Learning, note the model is not trained here, so this file does not take centuries to run.
    agent = DeepQLearningAgent.load(BOARD, "../policies/model.pt")

    # Uncomment the line below to train a new model from scratch, these are the setting used
    # agent = get_data_and_train(board, (10, 10), n_episodes=N_EPISODES00, batch_size=1024)

    deep_q_policy = agent.make_greedy_tabular_policy()

    data["Deep Q-Learning"] = collect_algorithm_data(BOARD, deep_q_policy, n_episodes=N_EPISODES, options=OPTIONS, seed=SEED, max_steps=MAX_STEPS)
    print("Finished collecting data for Deep Q-Learning (7/7)")

    print("_" * 100, "\n")

    # Plotting
    print("Creating Plots\n")

    # Reward over time
    print("Creating Reward over Time plots")
    plot_rewards_over_time(data, "../plots/r_over_time")

    # Total return comparisons
    print("Creating return comparison plot")
    plot_comparison_barchart(data, "../plots/comparison/total_return_comparison.png")
    
    # Histograms of returns
    print("Creating reward density plots")
    plot_reward_density(data, "../plots/return_probability")

    print("Finished first plot set")

    print("_" * 100, "\n")
    # Comparison plots
    print("Making comparison plot of MC, SARSA, and Q-Learning")
    mc_comparison_policy = MonteCarloPolicy(BOARD.observation_space, BOARD.action_space, seed=SEED)
    q_learning_comparison_policy = TemporalDifferencePolicy(BOARD.observation_space, BOARD.action_space, seed=SEED)
    sarsa_comparison_policy = TemporalDifferencePolicy(BOARD.observation_space, BOARD.action_space, seed=SEED)

    policy_funcs = (
        lambda n_episodes: mc_comparison_policy.first_visit_monte_carlo_control(BOARD, n_episodes=n_episodes, gamma=0.9,
                                                                                reset=False, epsilon=0.3),
        lambda n_episodes: q_learning_comparison_policy.q_learning(BOARD, n_episodes=n_episodes, alpha=0.5, gamma=0.9, reset=False),
        lambda n_episodes: sarsa_comparison_policy.sarsa(BOARD, n_episodes=n_episodes, alpha=0.5, gamma=0.9,
                                                         reset=False)
    )
    algorithms = ("Monte-Carlo", "Q-Learning", "SARSA")

    plot_reward_evolution_comparison(BOARD, ticks=COMPARISON_TICKS, n_samples=N_COMPARISON_SAMPLES,
                                     policy_train_functions=policy_funcs, algorithm_names=algorithms,
                                     plot_save_path="../plots/comparison/comparison_plot_final.png", options=OPTIONS, seed=SEED)

    print("Finished")