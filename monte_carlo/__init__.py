import random

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import agents
from game import Game, JobstGame


class MonteCarlo:


    def __init__(self, game: Game, strategy_set: dict,
                 agent_list: npt.NDArray[agents.Agent] = np.array([]),
                 generations: int = 100):

        self.game = game
        self.agent_list = agent_list
        self.strategy_set = strategy_set
        self.generations = generations
        self.agent_types = set([agent.label for agent in self.agent_list])
        self.reward_dict = {agent_type: 0 for agent_type in self.agent_types}
        self.agent_counts = {agent_type: np.array([count+1]) for agent_type, count in agents.Agent.agent_dict.items()}

    def __repr__(self):
        return(f"Number of agents: {len(self.agent_list)} \n"
               f"Generations: {self.generations} \n")

    def shuffle_population(self):
        random.shuffle(self.agent_list)

    def play_game(self, agent1: agents.Agent, agent2: agents.Agent):

        agent1.set_opponent(agent2)
        agent2.set_opponent(agent1)

        agent1.choose_strategy()
        agent2.choose_strategy()

        action1 = agent1.get_action()
        action2 = agent2.get_action()

        r1, r2 = self.game.get_reward(action1, action2)   # maybe want to have reward as an attribute for agent?
        self.reward_dict[agent1.label] += max(0, r1)
        self.reward_dict[agent2.label] += max(0, r2)

    def play_all_games(self):

        pairs = self.agent_list.reshape(len(self.agent_list)//2, 2)
        for pair in pairs:
            agent1, agent2 = pair[0], pair[1]
            self.play_game(agent1, agent2)

    def set_new_population(self):

        total_reward = sum(self.reward_dict.values())
        n_agents_total = len(self.agent_list)
        agents.Agent.agent_dict = {}  # reset class variable to empty
        self.agent_list = np.array([])
        for agent_type, r in self.reward_dict.items():
            n_agents = round((r/total_reward)*n_agents_total)
            if n_agents > 0:
                self.agent_list = np.append(self.agent_list,
                                            np.array([agents.Agent(agent_type,
                                                      self.strategy_set[agent_type]) for _ in range(n_agents)]))
            self.agent_counts[agent_type] = np.append(self.agent_counts[agent_type], n_agents)

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
            ax.plot(agent_counts/total_agents, label=agent_type)

        plt.title('Plot of agent populations over generations')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Agent type share of population')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    options = [0,1,2,3,4,5]
    satisfia_set = {'satisfia': {'actions': options,
                                 'probabilities': [0, 0, 0, 1, 0, 0]},
                    'maximiser': {'actions': options,
                                  'probabilities': [0, 1, 0, 0, 0, 0]}
                    }

    maximiser_set = {'satisfia': {'actions': options,
                                  'probabilities': [0, 1, 0, 0, 0, 0]},
                     'maximiser': {'actions': options,
                                   'probabilities': [0, 0, 0, 0, 0, 1]}
                     }

    BIG_STRATEGY_SET = {'satisfia': satisfia_set, 'maximiser': maximiser_set}  # we should make this into a table or smth
    satisfias = np.array([agents.Agent('satisfia', satisfia_set) for _ in range(720)])
    maximisers = np.array([agents.Agent('maximiser', maximiser_set) for _ in range(280)])
    AGENT_LIST = np.append(satisfias, maximisers)
    GAME = JobstGame
    GENERATIONS = 100
    mc = MonteCarlo(GAME, BIG_STRATEGY_SET, AGENT_LIST, GENERATIONS)

    agent_counts_final = mc.iterate_generations()
    if agent_counts_final['maximiser'] > agent_counts_final['satisfia']:
        print(True)
