# LSMemoryModel
A codebase that allows you to play with different online learning algorithms that model memory 

### How to install:
Run the following sequence of all commands
```
make virtulenv
source venv/bin/activate
make all
```
This should create a virtual env called venv, install all dependencies,
install the package and activate the created virtual environment.

### To find experiments, go to the scripts folder
```
cd LSMemoryModel/scripts
```

### To set constants (num_actions. num_contexts, epsilon) go to
```
cd LSMemoryModel/constants
vim discrete.py
```

# The codebase is divided into 7 folders
## Algos
Each Algo object is a learning system. It has some internal store of memory, as well the capacity to update this internal state at every time step. Based on some (or no) input to performs a computation on its internal state to produce a signal. Algo objects do not directly interact with the environment (env objects)
### Currently supported algorithms
- Bayesian Memory Model
- RBF Kernel based memory (bugs present)
- Loyd Leslie Based Bayesian Memory model
- Policy Gradient Method based memory

## Agents
An agent is a composition of one or many systems (algorithm objects). It recieves some contextual input from the environment, and relays all or part of this input to each of its systems (algos). Then based on the signals recieved from these algos objects, it performs a computation to chose a final action to implement. The Agent objects directly interact with the environment. In the current formulation, an agent has no memory of its own. It also doesn't have direct access to its algo's memories, but simply functions that return some form of a prediction signal.
### Currently Supported Agents
- Bayes Agent
- Loyd Leslie Agent (bugs present)
- Dual Rate Model Agent
- RBF Kernel agent
## Data
Data generation mechanisms are stored here. This could be simple scripts to fetch data, or generate synthetic deterministic datasetd, or declare stochastic data generation procedures to be accessed by an environment.
## Constants
Contains constants like important paths, magic numbers, restrictions filed under separate identifyable names.

## Envs
The environment object drives most of the code. Anenv object accesses some Data object (if needed) in order to setup its initial state. The environment initialised in conjunction with some agent. It updates its internal state based on the action provided by the agent, and returns a reward (or context) signal to the agent. Simulations are run via the train function thats defined in the env base class. Visual mode if enabled, allows us to view the agents policy at each time step, with the simulation run at a slower pace for visual ease.
### Currently available envs
- Discrete Environment: a simple finite context environment, where each context has only 1 rewarding action.
- Loyd Env (bugs present): Chinese Restaurant Proces as defined in the Loyd Leslie paper.

## Scripts
Driver scripts that run initialise environments with agents and other hyperparameters, and run the train function.
## Utils
Contains certain "pure" functions that aren't specific to any object, but are reusable pieces of code.
