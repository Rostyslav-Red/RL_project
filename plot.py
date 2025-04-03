import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# Load data
with open("src/output_QLearning.txt", "r") as file:
    qlearning_data = np.array([float(line.strip()) for line in file])[:500]

with open("src/output_SARSA.txt", "r") as file:
    sarsa_data = np.array([float(line.strip()) for line in file])[:500]

with open("src/output_MC.txt", "r") as file:
    mc_data = np.array([float(line.strip()) for line in file])[:500]

# Optional smoothing
qlearning_smooth = moving_average(qlearning_data)
sarsa_smooth = moving_average(sarsa_data)
mc_smooth = moving_average(mc_data)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sarsa_smooth, label="SARSA", color="crimson", linewidth=2)
plt.plot(qlearning_smooth, label="Q-learning", color="royalblue", linewidth=2)
plt.plot(mc_smooth, label="Monte Carlo", color="seagreen", linewidth=2)

plt.title("Episode Reward Over Time", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Total Reward", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(loc="lower right", fontsize=11)
plt.tight_layout()
plt.show()

# Print means
print("Q-learning mean reward:", np.mean(qlearning_data))
print("SARSA mean reward:", np.mean(sarsa_data))
print("Monte Carlo mean reward:", np.mean(mc_data))
