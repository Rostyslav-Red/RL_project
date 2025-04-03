# Reinforcement Learning Project
### By Ella Kemperman, Ioana-Anastasia Orasanu, Rostyslav Redchyts, and Mila van Rijsbergen
This project contains a variety of algorithms used in reinforcement learning, such as Policy Iteration, Value Iteration,
Monte-Carlo Methods, Q-Learning, SARSA, and Deep Q-Learning. These algorithms are used by an agent in a custom 
[gymnasium](https://gymnasium.farama.org/) environment, in which the agent is a cat attempting to catch a mouse. The cat
can climb on trees, and wait in the trees to pounce on the mouse when it comes close. There are also walls in the environment,
which the cat cannot get past, while the mouse is able to sneak under them. This dynamic environment is challenging enough
that simple random algorithms perform terrible in it, while more optimised algorithms can easily learn the environment.

## Git Repository
The git repository can be found at:
- https://github.com/Rostyslav-Red/RL_project

## Repository Structure
The repository consists of 3 main folders. The `src` folder contains the source code, with explanation about its structure
found below. The `plots` folder contains plots that can be used to compare performance of the different algorithms. The 
`policies` folder contains a set of pre-trained policies that can be loaded, as well as a pre-trained model for Deep Q-Learning
(stored in `policies/model.pt`), and the episode data used to construct that model (stored in `policies/episodes.json`).
The `report_src` folder contains the LaTeX source code used to construct the report.

### Code Structure
The 'src' folder contains the source code of  the project. The main files of note that can be run are `main` and `data_collection`. 
A graph of the classes created in src can be seen below:

<img src="https://github.com/Rostyslav-Red/RL_project/blob/main/plots/rl_project_uml.png?raw=true" alt="drawing" width="900"/>

The `constants.py` file is a configuration file that specifies the actions and their corresponding movement vectors that the
agent can take, it also defines the rewards that can be gotten at each cell.

The `Board` (`board.py`) class is the main environment class, which implements the environment in code. The environment itself is a 
grid world where an agent needs to chase a target, that moves independently. The goal of the agent is to catch this target
as quickly as possible. The `Board` is built out of `Cell` objects, found in `cell.py`.

The `Policy` (`policy.py`, `dynamic_programming_policy.py`, `monte_carlo_policy.py`,`temporal_difference_policy.py`) classes are the classes that handle the main Reinforcement Learning algorithms. The base class handles the common
functionality of all policies, such as generating all states possible based on an observation space, and mapping observations
to actions, that are either deterministic or not. This stochasticity is handled by the `ActionSampler` (`action_sampler.py`) class,
which assigns a probability to each action and can randomly sample one of these actions using the probabilities.

The subclasses of `Policy`, `DynamicProgrammingPolicy` (`dynamic_programming_policy.py`), `MonteCarloPolicy` (`monte_carlo_policy.py`),
and `TemporalDifferencePolicy` (`temporal_difference_policy.py`), each handle their specific algorithms, each implemented as
methods of these classes.

Finally, the `Agent`  (`agent.py`) abstract class and it's subclasses handle interaction with the environment. The `HumanAgent` allows 
one to control the actions taken in the environment via the console, the `RandomAgent` randomly chooses an action at each
point, the `PolicyAgent` uses a `Policy` to determine its actions, and the `DeepQLearningAgent` (`deep_qlearning.py`) 
uses a neural network to determine its actions.

## Setting Up
To make sure all dependencies are installed, run 
```shell
pip3 install -r requirements.txt
```

## Running the Code
There are two main ways to run the code. For running individual tests, please run the `main` file, which can be 
configured as explained below. The second way to run the code, running each algorithm at once and creating informative
plots of the resulting policies, is done by running the `data_collection` file. This file is ready to go to create
the plots used in the report.

### Running `data_collection`
The file `data_collection` creates the plots used in the report out of the box. No additional configuration is needed,
although it is always possible to change some of the parameters used and see what their impact is. 

### How to configure `main`
#### Configuring the environment
The gymnasium environment used in this project (a grid world with a cat that needs to catch a moving mouse) is already 
automatically constructed at the start of main. Compatibility with other environments is not guaranteed, although 
theoretically, the Temporal Difference Learning, Monte-Carlo, and Deep Q-Learning algorithms should work for all environments
that are discrete and episodic. Dynamic Programming only works if the unwrapped environment has attributes `.P` and `.R`,
where `.P` is a dictionary mapping state action to all possible successor states, and `.R` a dictionary mapping state to
reward.

It is possible to choose the starting position of both the agent and the target. This can be done by modifying the 
`options` variable (line 15). The value corresponding to the `cat_position` key (should be a numpy 2D array with values
$0 \leq x\leq 3$) specifies where the agent will spawn, while the `target_position` (same requirements as the `cat_position`)
states where the target will spawn. In line 20, in the `board.reset()` call, the keyword argument `options` should then be
set to the `options` variable, allowing the board to reset positions to the specified ones. Leaving `options` empty randomises
positions. The `seed` keyword argument can be set to make sure the stochastic elements of the environment behave the same across
different runs.

#### Choosing a policy
Policies are used by `PolicyAgent`, so that they know what action to take at which step. If using another agent, such as
the `RandomAgent`, `HumanAgent`, or `DeepQLearningAgent`, a policy does not need to be selected.

Creating a basic random policy:

This can be done simply by using:
```python
p = Policy(board.observation_space, board.action_space)
```

Methods of choosing a dynamic programming policy:

1. By loading a saved policy
    ```python
    p = Policy.load(board, "../policies/value_iteration_policy.json")
    ```

2. By specifying the 'algorithm' argument of the instance of DynamicProgrammingPolicy
    ```python
   p = DynamicProgrammingPolicy(
        board, algorithm="ValueIteration", discount=0.1, stopping_criterion=0.001
    )
   ```

3. By calling the 'find_policy' method on the instance of DynamicProgrammingPolicy
    ```python
    p = DynamicProgrammingPolicy(board).find_policy("ValueIteration")
    ```
    

Methods of choosing a Monte-Carlo policy:
1. By loading a saved policy
    ```python
    p = Policy.load(board, "../policies/monte_carlo_policy.json")
    ```
   
2. By specifying the 'algorithm' argument of the instance of TemporalDifferencePolicy, and the keyword arguments
   required.
    ```python
   p = MonteCarloPolicy(
       board.observation_space, board.action_space, algorithm="FirstVisitEpsilonGreedy", env=board, n_episodes=1000, gamma=0.9, epsilon=0.3
       )
    ```
   
3. By calling the corresponding method of the to-be-used algorithm on an instance of TemporalDifferencePolicy:
    ```python
    p = MonteCarloPolicy(board.observation_space, board.action_space).first_visit_monte_carlo_control(board, n_episodes=1000)
   ```

Methods of creating a Temporal Difference Learning policy:

1. By loading a saved policy
    ```python
    p = Policy.load(board, "../policies/td_qlearning.json")
    ```
   
2. By specifying the 'algorithm' argument of the instance of TemporalDifferencePolicy, and the keyword arguments
   required.
    ```python
   p = TemporalDifferencePolicy(
       board.observation_space, board.action_space, algorithm="QLearning", env=board, n_episodes=1000, alpha=0.5, gamma=0.9
       )
    ```
   
3. By calling the corresponding method of the to-be-used algorithm on an instance of TemporalDifferencePolicy:
    ```python
    p = TemporalDifferencePolicy(board.observation_space, board.action_space).q_learning(board, n_episodes=1000)
   ```
    
#### Creating an agent
There are several types of agents, such as the `PolicyAgent`, `RandomAgent`, `HumanAgent`, or `DeepQLearningAgent`.

A `PolicyAgent` acts according to a policy. A `PolicyAgent` can be constructed using:
```python
agent = PolicyAgent(board, p)
```
With `p` the policy used.

`RandomAgents` act completely randomly. They can be constructed using:
```python
agent = RandomAgent(board, seed=None)
```

A `HumanAgent` can be controlled via the console. They can be constructed using:
```python
agent = HumanAgent(board)
```

Finally, there is the `DeepQLearningAgent`. This samples actions by greedily acting on a value function which is learnt
by a neural network. Methods of creating a `DeepQLearning` agent:
1. By loading saved model weights
    ```python
    agent = DeepQLearningAgent.load(board, "../policies/model.pt")
    ```

2. By training a new model on a dataset. Data can be loaded from sampled data. Note there are many customisation 
   options for training, read the documentation for those.
     ```python
    data = RLData.load("../policies/episodes.json")
    agent = DeepQLearningAgent.build_model(board, (hidden_layer1, hidden_layer2, ...))
    agent.train(n_epochs=10, data=data, **kwargs)
    ```
    
3. By training a new model on a newly sampled dataset. Note there are many customisation options for training, 
   read the documentation for those.
    ```python
    agent = get_data_and_train(board, (hidden_layer1, hidden_layer2, ...), **kwargs)
   ```

#### Running an episode
Code for running an episode is already included in `main`. This calls `agent.run_agent(obs)`, which returns the final return
(sum of rewards) obtained over the course of the episode.