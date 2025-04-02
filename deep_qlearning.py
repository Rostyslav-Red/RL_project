import json
from typing import Optional, Iterable
from torch.utils.data import DataLoader
from agent import Agent, PolicyAgent
from copy import deepcopy
from functools import reduce
from itertools import pairwise
from operator import add
import gymnasium as gym
from gymnasium.core import ObsType
import torch
from torch import nn
from policy import Policy
import numpy as np
from action_sampler import ActionSampler

class DeepQLearningAgent(Agent):
    """
    This agent implements the Deep Q-learning algorithm, using a neural network to approximate a value function,
    acting greedily on this value function.
    """

    def __init__(self, env: gym.Env, model: nn.Module):
        """
        Constructs the Deep Q-learning agent. Don't use this constructor, the factory methods are the preferred way
        of constructing a Deep Q-Learning agent. Use `build_model()`, `load()`, or `get_data_and_train()` instead.

        @param env: The environment the agent will act on.
        @param model: The nn.Module used to approximate the value function. Should have an input layer that matches the
                      observation space of the environment and an output layer that matches the action space of the
                      environment.
        """
        super().__init__(env)
        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__model = model.to(self.__device)

    @property
    def model(self):
        return self.__model

    def train(self, n_epochs: int, data: 'RLData', *, batch_size = 64, gamma: float = 0.9, retarget: int = 10, lr=0.01):
        """
        Trains the network contained in the Deep Q-Learning agent using batch gradient descent.

        @param n_epochs: Number of times the full dataset will be trained on.
        @param data: The set of (State, Action, Reward, State, Terminal) the dataset will be trained on.
        @param batch_size: The size of the batches trained on.
        @param gamma: The discount.
        @param retarget: Amount of batches after which the target model will be updated.
        @param lr: Learning rate.
        """
        # Initialise everything needed for training.
        self.__model.train()
        optimiser = torch.optim.SGD(self.__model.parameters(), lr=lr)

        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        target_model = deepcopy(self.__model)

        # Train for n_epochs
        for i in range(n_epochs):
            print(f"Epoch {i + 1}/{n_epochs}")

            t = 0  # Keeps track of when network retargets.

            # Main training loop.
            for obs, action, reward, new_obs, ended in loader:
                optimiser.zero_grad()

                # Make sure everything is on the correct device and in the correct form.
                obs = torch.tensor(np.array(obs)).to(self.__device).float().T
                new_obs = torch.tensor(np.array(new_obs)).to(self.__device).float().T
                reward = reward.to(self.__device)
                action = action.to(self.__device)
                ended = ended.to(self.__device)

                # Forward evaluations.
                qs = self.__model.forward(obs)
                target_qs = target_model.forward(new_obs)

                # Compute loss
                target = (reward + gamma * torch.max(target_qs, dim=1)[0] * (1 - ended)).detach()
                loss = ((target - qs.gather(1, action.unsqueeze(1)).squeeze(1)) ** 2).mean()

                # Find gradient and take step in correct direction.
                loss.backward()
                optimiser.step()

                # Retargeting when t reaches correct values.
                t += 1
                if t % retarget == 0:
                    target_model.load_state_dict(self.__model.state_dict())

        self.__model.eval()

    def _generate_move_helper(self) -> int:
        """
        Generates a move by acting greedily on the value function of the model.

        @return: An action to be taken in the environment in this state.
        """
        with torch.no_grad():
            obs = self._obs_to_tensor(self._obs)
            return int(self.__model.forward(obs).argmax())

    def _obs_to_tensor(self, obs: ObsType) -> torch.Tensor:
        """
        Converts an observation from the environment to a torch.Tensor.

        @param obs: Observation to be converted.
        @return: Converted observation.
        """
        return torch.Tensor(tuple(map(int, gym.spaces.flatten(self._obs_space, obs)))).to(self.__device)

    def make_greedy_tabular_policy(self) -> Policy:
        """
        @return: A greedy tabular policy based on the internal model of the agent.
        """
        keys = Policy.get_keys(self._obs_space)
        policy_dict = {state:
                           ActionSampler.deterministic_action_sampler(int(self.model.forward(torch.tensor(state).to(self.__device).float()).argmax()))
                       for state in keys}
        return Policy(self._obs_space, self._action_space, policy=policy_dict)

    def save(self, file_name: str):
        torch.save(self.__model, file_name)

    @staticmethod
    def build_model(env: gym.Env, layer_sizes: tuple[int, ...]) -> 'DeepQLearningAgent':
        """
        Builds a Deep Q-learning agent that acts in the specified environment. By default the model used is a stack
        of linear layers.

        @param env: The environment the agent will act in.
        @param layer_sizes: The sizes of the hidden layers of the network. First value is the first hidden layer size,
                            second value is the second, etc.
        @return: A DeepQLearningAgent with the specified model ready to be trained.
        """
        # Determine size of input and output layer.
        input_size = len(gym.spaces.flatten_space(env.observation_space).low)
        output_size = int(env.action_space.n)

        # Create pairs of tuples for creating the linear layers, create a tuple of these layers with ReLU in between.
        layer_sizes = (input_size,) + layer_sizes + (output_size,)
        layers = tuple(reduce(add, map(lambda x: (nn.Linear(x[0], x[1]), nn.ReLU()), pairwise(layer_sizes))))

        # Construct model, :-1 excludes the final ReLU layer, since output q-values can also be negative.
        model = nn.Sequential(*layers[:-1])

        return DeepQLearningAgent(env, model)

    @staticmethod
    def load(environment: gym.Env, file_name: str) -> 'DeepQLearningAgent':
        """
        Loads a Deep Q-Learning agent from a set of model weights.

        @param environment: The environment the Agent will act in.
        @param file_name: The file name of the file containing the model weights.
        @return: A DeepQLearningAgent constructed from the specified model weights.
        """
        model = torch.load(file_name, weights_only=False)
        model.eval()
        return DeepQLearningAgent(environment, model)


class RLData(torch.utils.data.Dataset):
    """
    Dataset used by DeepQLearningAgents to train. Consists of samples from many episodes, each sample taking the form:
    (State, Action, Reward, State, Terminal)
    """

    def __init__(self, observations: tuple[tuple[int, ...], ...], actions: tuple[int, ...], rewards: tuple[int, ...],
                 new_observations: tuple[tuple[int, ...]], ended: tuple[int]):
        """
        Constructs the dataset. Don't use this constructor, the `sample_data()` and `load()` factory methods are
        preferable over this constructor.

        @param observations: The observations in tuple form from the environment.
        @param actions: The action taken from the corresponding observation.
        @param rewards: The reward gotten after taking the action.
        @param new_observations: The new state after taking the action from the observation.
        @param ended: Is the new state terminal or not.
        """
        assert len(observations) == len(actions) == len(rewards) == len(new_observations) == len(ended), \
            "All input data must have the same length."
        self.__observations = observations
        self.__actions = actions
        self.__rewards = rewards
        self.__new_observations = new_observations
        self.__ended = ended


    def __len__(self):
        """
        Compatibility with torch.utils.data.Dataset, gives length of dataset.
        """
        return len(self.__observations)

    def __getitem__(self, i) -> tuple[tuple[int, ...], int, float, tuple[int, ...], int]:
        """
        Needed for compatibility with torch.utils.data.Dataset, gives data found at index i.

        @param i: Index.
        @return: Tuple of form (State, Action, Reward, State, Terminal).
        """
        return self.__observations[i], self.__actions[i], self.__rewards[i], self.__new_observations[i], self.__ended[i]

    @staticmethod
    def sample_data(env: gym.Env, n_episodes: int, policy: Optional[Policy] = None, seed: Optional[int] = None) -> 'RLData':
        """
        Creates a dataset to be used for Deep Q-Learning training by sampling many episodes.

        @param env: The environment from which data will be sampled.
        @param n_episodes: Number of episodes that will be sampled.
        @param policy: Policy used for sampling, by default a policy assigning equal probability to all actions.
        @param seed: Random seed used for sampling.
        @return: A dataset ready for use in DeepQLearningAgent training.
        """
        policy = policy if policy else Policy(env.observation_space, env.action_space, seed=seed)
        observations, actions, rewards, new_observations, ended = (), (), (), (), ()

        for i in range(n_episodes):
            obs, acts, rs, s_primes, terminated = PolicyAgent.sample_episode(env, policy, seed=seed)
            observations += obs
            actions += acts
            rewards += rs
            new_observations += s_primes
            ended += terminated

            if i % 1000 == 0:
                print(f"Sampled {i} episodes")
        print(len(observations))
        return RLData(observations, actions, rewards, new_observations, ended)

    def save(self, file_name: str):
        """
        Saves the dataset to a file in json format, so sampling does not need to be repeated.

        @param file_name: File the dataset will be stored in.
        """
        # Put data in dict so json.dumps can handle it.
        data_dict = {
            "observations": self.__observations,
            "actions": self.__actions,
            "rewards": self.__rewards,
            "new_observations": self.__new_observations,
            "ended": self.__ended
        }
        with open(file_name, "w") as f:
            f.write(json.dumps(data_dict))

    @staticmethod
    def load(file_name: str) -> 'RLData':
        """
        Constructs a dataset for DeepQLearningAgent training from a saved RLData dataset.

        @param file_name: File in which the data is stored.
        @return: A dataset ready for use in DeepQLearningAgent training.
        """
        with open(file_name, "r") as f:
            data_dict = json.loads(f.read())

        # Make sure everything is tuple
        data_dict["observations"] = tuple(map(tuple, data_dict["observations"]))
        data_dict["actions"] = tuple(data_dict["actions"])
        data_dict["rewards"] = tuple(data_dict["rewards"])
        data_dict["new_observations"] = tuple(map(tuple, data_dict["new_observations"]))
        data_dict["ended"] = tuple(data_dict["ended"])

        return RLData(**data_dict)


def get_data_and_train(env: gym.Env, layer_sizes: tuple[int, ...], *, episodes_save_file: Optional[str] = None,
                       model_save_file: Optional[str] = None, n_episodes: int = 1000, retarget: int = 10,
                       batch_size: int = 64, sample_policy: Optional[Policy] = None, seed: Optional[int] = None,
                       n_epochs: int = 10, gamma: float = 0.9, lr: float = 0.01) -> DeepQLearningAgent:
    """
    Samples episodes and trains a DeepQLearningAgent on these episodes, returning the trained agent.

    @param env: The environment the agent will be trained on.
    @param layer_sizes: The sizes of the hidden layers the agent will have.
    @param episodes_save_file: Optional, the file the sampled episode data will be saved to.
    @param model_save_file: Optional, the file the trained model will be saved to.
    @param n_episodes: Number of episodes sampled, default is 1000.
    @param retarget: Amount of batches after which the target model will be updated. Default is 10.
    @param batch_size: Batch size used in training, default is 64.
    @param sample_policy: Sample policy used for generating data, default is a uniform policy over all actions.
    @param seed: Optional, random seed used for sampling.
    @param n_epochs: Number of epochs used in training, default is 10.
    @param gamma: Discount used, defaults to 0.9.
    @param lr: Learning rate, defaults to 0.01.
    @return: A trained DeepQLearningAgent.
    """
    # Generate episode data
    data = RLData.sample_data(env, n_episodes=n_episodes, policy=sample_policy, seed=seed)
    if episodes_save_file:
        data.save(episodes_save_file)

    # Create Neural Network, train and save it
    agent = DeepQLearningAgent.build_model(env, layer_sizes=layer_sizes)
    agent.train(n_epochs, data, retarget=retarget, batch_size=batch_size, gamma=gamma, lr=lr)

    if model_save_file:
        agent.save(model_save_file)

    # Load DeepQLearningAgent from weights
    return agent