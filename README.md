# SatisfIA - Multi-Agent 
## Top Level Summary
This repository provides object orientated Python code to explore the role of social network structures & several other relevant factors, on the survival or persistence of cohorts of different type of agent. 
This work has been undertaken as part of the Supervised Program for Alignment Research, and as part of the wider SatisfIA research group. This research group aims to establish agentic AI systems that do not maximise
for a given goal, avoiding the costly and dangerous results of ill-specified goals. This specific work explores competition between such agents and optimisers. 


## Installation
To install the package, clone the repository and run the following command in the root directory of the repository:
```bash
pip install -r requirements.txt
```

## Usage

### Run a simulation
To run a simulation, run the following command in the root directory of the repository:
```bash
python main.py
```

### Parameter searcch
To run a parameter search, run the following command in the root directory of the repository:
```bash
python iterate_model_params.py \
        -gT GRAPH_TYPE \
        -nA N_AGENTS \
        -sS SATSISFIA_SHARE \
        -gP P_PLAY_GAME \
        -lP P_SOCIAL_LEARNING \
        -lA LEARN_PARAM_A \
        -lB LEARN_PARAM_B
```

## Sub components: 
### Game
This class defines a basic normal form game object, currently only for 2 player normal form games, where payoffs are stored in the cells of an array. 

### Agent
This class defines an agent that is able to interact with a game object. 

### MonteCarlo
This defines a class of methods and structures to simulate many agents interacting over a series of generations. This enables the trajectory of a population of SatisfIA or Maximiser agents to be simulated. 

### Network 
This class inherits from the MonteCarlo class, but enables interactions between agents to take place on a social network structure. Different graph types can be provided to explore population trajectories on different 
networks such as Barabasi-Albert, Fully connected or Watts-Strogatz, to name a few. 
