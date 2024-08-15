import random
from typing import List, Type

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Patch
import logging

from agents import SatisfiaAgent, MaximiserAgent, SATISFIA_SET, MAXIMISER_SET, Agent
from games import Game, JOBST_GAME
from monte_carlo import MonteCarlo, combined_strategies

logger = logging.getLogger(__name__)


def gen_agent_population(n_agents: int, satisfia_share: float) -> npt.NDArray[Agent]:
    satisfia_count, maximiser_count = round(n_agents*satisfia_share), round(n_agents*(1-satisfia_share))

    satisfias = np.array([SatisfiaAgent(SATISFIA_SET) for _ in range(satisfia_count)])
    maximisers = np.array([MaximiserAgent(MAXIMISER_SET) for _ in range(maximiser_count)])
    agent_population = np.append(satisfias, maximisers)
    # np.random.shuffle(agent_population)
    return agent_population


class SatisfiaMaximiserNetwork(MonteCarlo):

    def __init__(self,
                 game: Game,
                 strategy_dict: dict,
                 satisfia_share: float,
                 generations: int,
                 base_graph: nx.Graph,
                 draw_network_interval: int,
                 learn_param_a: float = 0.5,
                 learn_param_b: float = 0.5):

        self.n_agents = len(base_graph.nodes())
        agent_list = gen_agent_population(self.n_agents, satisfia_share)
        super().__init__(game, strategy_dict, agent_list, generations)
        self.satisfia_share = satisfia_share
        self.graph = self.initialize_graph(base_graph.copy())
        self.draw_network_interval = draw_network_interval
        self.learn_param_a = learn_param_a
        self.learn_param_b = learn_param_b
        self.closeness_centrality = {agent_type: [self.get_avg_closeness_centrality(agent_type)]
                                     for agent_type in self.agent_types}

    @property
    def nodes(self):
        return np.array(self.graph.nodes(data=True))

    def set_agent_by_id(self, id: int, new_agent: Agent):
        assert id == new_agent.id, "New agent must have the given ID as an attribute"
        for i, agent in enumerate(self.agent_list):
            if agent.id == id:
                self.agent_list[i] = new_agent
                self.graph.nodes[i]['data'] = new_agent
                return
        raise ValueError(f"Can't update agent: ID {id} not found in agent list.")

    def initialize_graph(self, base_graph: nx.Graph):
        np.random.shuffle(self.agent_list)
        for i, agent in enumerate(self.agent_list):
            base_graph.nodes[i]['data'] = agent
        return base_graph

    def get_random_edge(self):
        edges = np.array(self.graph.edges)
        edge_idx = np.random.randint(len(edges))
        chosen_edge = edges[edge_idx, :]
        return chosen_edge

    def get_random_node(self):
        node_idx = np.random.randint(self.n_agents)
        return node_idx

    def get_random_neighbor(self, node):
        neighbors = self.graph.neighbors(node)
        neighbor = random.choice([n for n in neighbors])
        return neighbor

    def node_social_learning(self, learner_agent: Agent, neighbor_agent: Agent) -> None:

        p_switch = 1/(1+np.exp(self.learn_param_a + self.learn_param_b*(learner_agent.payoff - neighbor_agent.payoff)))
        r = np.random.uniform()

        if r < p_switch:

            logger.debug("--------------------------------------------------------------------")
            logger.debug(f"Switch between: Learner {learner_agent} with {learner_agent.payoff}"
                  f" and Neighbor {neighbor_agent} with {neighbor_agent.payoff} payoff")
            learner_agent = learner_agent.transmute(neighbor_agent)
            logger.debug(f"new learner: {learner_agent}, new neighbor: {neighbor_agent}")

            self.set_agent_by_id(learner_agent.id, learner_agent)

    def play_game_process(self):
        node1, node2 = self.get_random_edge()
        agent1 = self.graph.nodes[node1]['data']
        agent2 = self.graph.nodes[node2]['data']
        self.play_game(agent1, agent2)

    def social_learning_process(self):
        learner_node = self.get_random_node()
        neighbor_node = self.get_random_neighbor(learner_node)

        learner_agent = self.graph.nodes[learner_node]['data']
        neighbor_agent = self.graph.nodes[neighbor_node]['data']

        if neighbor_agent.type != learner_agent.type:
            self.node_social_learning(learner_agent, neighbor_agent)

    def draw_network(self, generation: int):
        color_map = [agent.type.color for agent in self.agent_list]
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(7, 7))
        nx.draw_networkx_nodes(self.graph, pos, node_color=color_map, node_size=30)
        nx.draw_networkx_edges(self.graph, pos)
        nx.draw_networkx_labels(
            self.graph,
            pos={node: (coords[0], coords[1] + 0.05) for node, coords in pos.items()}
        )
        plt.title(f"Network (generation {generation})")
        plt.legend(handles=[Patch(facecolor=agent_type.color, edgecolor='black', label=agent_type.__name__)
                            for agent_type in self.agent_types])
        plt.show()

    def iterate_generations(self, p_play_game: float, p_social_learning: float, **kwargs):
        for i_gen in range(self.generations):
            if random.random() < p_play_game:
                self.play_game_process()
            if random.random() < p_social_learning:
                self.social_learning_process()
            # Todo: rewiring

            self.store_agent_counts()
            self.store_avg_closeness_centrality()
            if kwargs['plot'] and i_gen % self.draw_network_interval == 0:
                # self.draw_network(i_gen)
                pass

        logger.debug(kwargs)
        if kwargs['plot']:
            if 'fig' in kwargs:
                super().plot_agent_counts(**kwargs)
            else:
                self.plot_agent_counts()
            # self.plot_average_centrality()

    def count_internal_edges(self, nodes):
        subgraph = self.graph.subgraph(nodes)
        return subgraph.number_of_edges()

    def count_external_edges(self, nodes_a, nodes_b):
        count = 0
        for node in nodes_a:
            count += len([neighbor for neighbor in self.graph.neighbors(node) if neighbor in nodes_b])
        return count

    def get_avg_closeness_centrality(self, agent_type: Type[Agent] = None):
        if agent_type is None:
            return nx.closeness_centrality(self.graph)
        else:
            closeness_centralities = [nx.closeness_centrality(self.graph, u=node_key)
                                      for node_key, node_data in self.graph.nodes(data=True)
                                      if node_data['data'].type == agent_type]
            return np.array(closeness_centralities).mean()

    def get_avg_degree(self, agent_types=None):
        agent_types = self.agent_types if agent_types is None else agent_types
        degrees = [val for key, val in dict(self.graph.degree()).items()
                   if self.graph.nodes[key]['data'].type in agent_types]
        avg_deg = sum(degrees)/len(degrees)
        return avg_deg

    def store_avg_closeness_centrality(self):
        for agent_type in self.agent_types:
            self.closeness_centrality[agent_type].append(self.get_avg_closeness_centrality(agent_type))

    def plot_agent_counts(self):
        for agent_type, counts in self.agent_counts.items():
            plt.plot(np.array(counts) / self.n_agents, label=agent_type.__name__)
        plt.title('Plot of agent populations over generations')
        plt.xlabel('Generations')
        plt.ylabel('Population')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()

    def plot_average_centrality(self):
        for agent_type in self.agent_types:
            plt.plot(self.closeness_centrality[agent_type], label=agent_type.__name__)
        plt.title('Plot of average closeness centrality of nodes by agent type')
        plt.xlabel('Generations')
        plt.ylabel('Average closeness centrality')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    N_AGENTS = 30
    EDGES_PER_NODE = 2
    BASE_BARABASI = nx.barabasi_albert_graph(N_AGENTS, EDGES_PER_NODE)

    BASE_FULL = nx.complete_graph(N_AGENTS)

    logging.basicConfig(level=logging.INFO)
    my_graph = SatisfiaMaximiserNetwork(
        JOBST_GAME,
        combined_strategies,
        0.65,
        60,
        BASE_BARABASI,
        50
    )
    my_graph.iterate_generations(1, 0.5, plot=True)
    my_graph.get_avg_closeness_centrality(MaximiserAgent)

