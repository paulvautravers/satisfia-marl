import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import agents
from agents import SatisfiaAgent, MaximiserAgent
from game import Game, JobstGame


class MonteCarlo:

    def __init__(self, game: Game, strategy_set: dict,
                 agent_list: npt.NDArray[agents.Agent] = np.array([]),
                 generations: int = 100):

        self.game = game
        self.agent_list = agent_list
        self.strategy_set = strategy_set
        self.generations = generations
        self.agent_types = set([agent.type for agent in self.agent_list])
        self.reward_dict = defaultdict(int)
        self.agent_counts = {agent_type: np.array([count+1]) for agent_type, count in agents.Agent.agent_dict.items()}

    def __repr__(self):
        return(f"Number of agents: {len(self.agent_list)} \n"
               f"Generations: {self.generations} \n")

    def shuffle_population(self):
        random.shuffle(self.agent_list)

    def play_game(self, agent1: agents.Agent, agent2: agents.Agent):

        action1 = agent1.get_action(agent2)
        action2 = agent2.get_action(agent1)

        r1, r2 = self.game.get_reward(action1, action2)   # maybe want to have reward as an attribute for agent?

        self.reward_dict[type(agent1)] = max(0, self.reward_dict[type(agent1)] + r1)
        self.reward_dict[type(agent2)] = max(0, self.reward_dict[type(agent2)] + r2)


    def play_all_games(self):

        pairs = self.agent_list.reshape(len(self.agent_list)//2, 2)
        for pair in pairs:
            agent1, agent2 = pair[0], pair[1]
            self.play_game(agent1, agent2)

    def set_new_population(self):

        total_reward = sum(self.reward_dict.values())
        n_agents_total = len(self.agent_list)
        agents.Agent.agent_dict = defaultdict(int)  # reset class variable to empty
        self.agent_list = np.array([])
        for agent_type, r in self.reward_dict.items():
            n_agents = round((r/total_reward)*n_agents_total)
            if n_agents > 0:
                self.agent_list = np.append(self.agent_list,
                                            np.array([agent_type(self.strategy_set[agent_type])
                                                      for _ in range(n_agents)]))
            self.agent_counts[agent_type] = np.append(self.agent_counts[agent_type], n_agents)  ## Redundant with the agent_dict of the Agent class

    def iterate_generations(self, plot: bool = False):

        for g in range(self.generations):
            self.shuffle_population()
            self.play_all_games()
            self.set_new_population()

        if plot:
            self.plot_agent_counts()

        agent_counts_final = {agent_type: counts[-1] for agent_type, counts in self.agent_counts.items()}
        return agent_counts_final

    def plot_agent_counts(self):

        fig, ax = plt.subplots(figsize=(10, 10))
        total_agents = len(self.agent_list)
        for agent_type, agent_counts in self.agent_counts.items():
            ax.plot(agent_counts/total_agents, label=agent_type.__name__)

        plt.title('Plot of agent populations over generations')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Agent type share of population')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    options = [0,1,2,3,4,5]
    satisfia_set = {SatisfiaAgent: {'actions': options,
                                 'probabilities': [0, 0, 0, 1, 0, 0]},
                    MaximiserAgent: {'actions': options,
                                  'probabilities': [0, 1, 0, 0, 0, 0]}
                    }

    maximiser_set = {SatisfiaAgent: {'actions': options,
                                  'probabilities': [1, 0, 0, 0, 0, 0]},
                     MaximiserAgent: {'actions': options,
                                   'probabilities': [0, 0, 0, 0, 0, 1]}
                     }

    BIG_STRATEGY_SET = {SatisfiaAgent: satisfia_set, MaximiserAgent: maximiser_set}  # we should make this into a table or smth
    satisfias = np.array([agents.SatisfiaAgent(satisfia_set) for _ in range(73)])
    maximisers = np.array([agents.MaximiserAgent(maximiser_set) for _ in range(27)])
    AGENT_LIST = np.append(satisfias, maximisers)
    GAME = JobstGame
    GENERATIONS = 100
    mc = MonteCarlo(GAME, BIG_STRATEGY_SET, AGENT_LIST, GENERATIONS)

    agent_counts_final = mc.iterate_generations(plot=True)
    if agent_counts_final[MaximiserAgent] > agent_counts_final[SatisfiaAgent]:
        print(True)
