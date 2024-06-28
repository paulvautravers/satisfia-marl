import random

import numpy as np
import numpy.typing as npt

import agents
from game import Game


class MonteCarlo:

    def __init__(self, game: Game, agent_list: npt.NDArray[agents.Agent] = np.array([]),
                 generations: int = 0):

        self.game = game
        self.agent_list = agent_list
        self.generations = generations
        self.agent_types = set([agent.label for agent in self.agent_list])
        self.reward_dict = {agent_type: 0 for agent_type in self.agent_types}

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

        self.reward_dict[agent1.label] += r1
        self.reward_dict[agent2.label] += r2

    def play_all_games(self):

        pairs = self.agent_list.reshape(len(self.agent_list)//2, 2)
        for pair in pairs:
            agent1, agent2 = pair[0], pair[1]
            self.play_game(agent1, agent2)

    def set_new_population(self):

        total_reward = sum(self.reward_dict.values())
        n_agents_total = len(self.agent_list)
        agent_pop_dict = {agent_type: [] for agent_type in self.agent_types}
        agents.Agent.agent_dict = {}  # reset class variable to empty
        for agent_type, r in self.reward_dict:
            n_agents = int((r/total_reward)*n_agents_total)
            if n_agents > 0:
                agent_pop_dict[agent_type] = [agents.Agent(agent_type) for _ in range(n_agents)]
