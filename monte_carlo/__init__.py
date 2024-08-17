import random
from collections import defaultdict
from typing import Callable, Concatenate, TypeVar, ParamSpec

Param = ParamSpec("Param")
RetType = TypeVar("RetType")


import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import agents
from agents import Agent, SatisfiaAgent, MaximiserAgent, SATISFIA_SET, MAXIMISER_SET
from games import Game, JOBST_GAME


class MonteCarlo:

    def __init__(self, game: Game, strategy_set: dict,
                 agent_list: npt.NDArray[agents.Agent],
                 generations: int = 100):

        #These attributes remain unchanged
        self.game = game
        self.strategy_set = strategy_set
        self.generations = generations

        self.initial_agent_list = agent_list
        self.agent_list, self.agent_types, self.reward_dict, self.agent_counts = None, None, None, None
        self.setup_agent_list_attributes()

    def setup_agent_list_attributes(self):

        self.agent_list = self.initial_agent_list
        self.agent_types = set([agent.type for agent in self.agent_list])
        self.reward_dict = defaultdict(int)

        initial_agent_counts = self.get_current_agent_counts()
        self.agent_counts = {agent_type: [initial_agent_counts[agent_type]] for agent_type in self.agent_types}

    def get_current_agent_counts(self):
        current_agent_counts = {agent_type: 0 for agent_type in self.agent_types}
        for agent in self.agent_list:
            current_agent_counts[agent.type] += 1
        return current_agent_counts

    def store_agent_counts(self):
        current_agent_counts = self.get_current_agent_counts()
        for agent_type, count_list in self.agent_counts.items():
            self.agent_counts[agent_type].append(current_agent_counts[agent_type])


    def set_agent_by_id(self, id: int, new_agent: Agent):
        assert id == new_agent.id, "New agent must have the given ID as an attribute"
        for i, agent in enumerate(self.agent_list):
            if agent.id == id:
                self.agent_list[i] = new_agent
                return
        raise ValueError(f"Can't update agent: ID {id} not found in agent list.")


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


        agent1.payoff = agent1.get_new_avg_payoff(max(0, r1))
        agent2.payoff = agent2.get_new_avg_payoff(max(0, r2))

    def play_all_games(self):

        pairs = self.agent_list.reshape(len(self.agent_list)//2, 2)
        for pair in pairs:
            agent1, agent2 = pair[0], pair[1]
            self.play_game(agent1, agent2)

    def set_new_population(self):

        total_reward = sum(self.reward_dict.values())
        n_agents_total = len(self.agent_list)

        Agent.reset()

        self.agent_list = np.array([])
        if total_reward > 0:
            for agent_type, r in self.reward_dict.items():

                n_agents = round((r/total_reward)*n_agents_total)
                self.agent_counts[agent_type].append(n_agents)
                if n_agents > 0:
                    self.agent_list = np.append(self.agent_list,
                                                np.array([agent_type(self.strategy_set[agent_type])
                                                          for _ in range(n_agents)]))

    def simulation_repeat_wrapper(self, n_repeats: int = 3) -> Callable[[Callable[Param, RetType]], Callable[Param, dict]]:

        def decorator(func: Callable) -> Callable[Param, dict]:

            def wrapper(*args: Param.args, **kwargs: Param.kwargs) -> dict:
                print("Inside wrapper")
                maximiser_counts_arr = np.empty(shape=(n_repeats, self.generations))
                satisfia_counts_arr = np.empty(shape=(n_repeats, self.generations))

                for i in range(n_repeats):
                    agent_counts = func(*args, **kwargs)

                    maximiser_counts_arr[i] = agent_counts[MaximiserAgent]
                    satisfia_counts_arr[i] = agent_counts[SatisfiaAgent]
                    self.setup_agent_list_attributes()

                maximiser_stats = np.percentile(maximiser_counts_arr, (25, 50, 75), axis=0)
                satisfia_stats = np.percentile(satisfia_counts_arr, (25, 50, 75), axis=0)

                return {MaximiserAgent: maximiser_stats, SatisfiaAgent: satisfia_stats}

            return wrapper

        return decorator



    def iterate_generations(self, **kwargs: Param.kwargs) -> dict:

        for g in range(self.generations - 1):
            self.shuffle_population()
            self.play_all_games()
            self.set_new_population()

        self.plot_agent_counts(**kwargs)

        return self.agent_counts

    def plot_agent_counts(self, **kwargs):
        if kwargs['plot'] == True:
            fig, ax = kwargs['fig'], kwargs['ax']
            total_agents = len(self.agent_list)

            for agent_type, agent_counts in self.agent_counts.items():
                if 'agent_to_plot' not in kwargs.keys():
                    ax.plot(np.array(agent_counts)/total_agents, label=agent_type.__name__)
                elif kwargs['agent_to_plot'] == agent_type:
                    ax.plot(np.array(agent_counts) / total_agents, label=agent_type.__name__, c='b', alpha=0.3)

            plt.title('Plot of agent populations over generations')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Agent type share of population')



# MonteCarlo globals
combined_strategies = {SatisfiaAgent: SATISFIA_SET, MaximiserAgent: MAXIMISER_SET}  # we should make this into a table or smth


if __name__ == '__main__':

    satisfias = np.array([agents.SatisfiaAgent(SATISFIA_SET) for _ in range(15)])
    maximisers = np.array([agents.MaximiserAgent(MAXIMISER_SET) for _ in range(15)])
    agent_population = np.append(satisfias, maximisers)
    generations = 20
    mc = MonteCarlo(JOBST_GAME, combined_strategies, agent_population, generations)


    mc2 = MonteCarlo(JOBST_GAME, combined_strategies, agent_population, generations)
    fig, ax = plt.subplots()
    agent_counts_pcs = mc2.simulation_repeat_wrapper(n_repeats=100)(mc2.iterate_generations)(plot=True, fig=fig,ax=ax)
    print(agent_counts_pcs)

    plt.show()
