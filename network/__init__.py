import random
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt

import agents
from agents import SatisfiaAgent, MaximiserAgent, SATISFIA_SET, MAXIMISER_SET
from games import Game, JOBST_GAME
from monte_carlo import MonteCarlo, combined_strategies


def gen_agent_population(n_agents: int, satisfia_share: float) -> npt.NDArray[agents.Agent]:
    satisfia_count, maximiser_count = round(n_agents*satisfia_share), round(n_agents*(1-satisfia_share))

    satisfias = np.array([agents.SatisfiaAgent(SATISFIA_SET) for _ in range(satisfia_count)])
    maximisers = np.array([agents.MaximiserAgent(MAXIMISER_SET) for _ in range(maximiser_count)])
    agent_population = np.append(satisfias, maximisers)
    np.random.shuffle(agent_population)
    return agent_population


class SatisfiaMaximiserNetwork(MonteCarlo):

    def __init__(self,
                 game: Game,
                 strategy_dict: dict,
                 n_agents: int,
                 satisfia_share: float,
                 generations: int,
                 m_edges_per_node: int,
                 graph_type: callable,
                 draw_network_interval: int):

        agent_list = gen_agent_population(n_agents, satisfia_share)
        super().__init__(game, strategy_dict, agent_list, generations)
        self.n_agents = n_agents
        self.satisfia_share = satisfia_share
        self.m_edges_per_node = m_edges_per_node
        self.satisfia_share = satisfia_share
        self.graph_type = graph_type
        self.color_map = []
        self.graph = self.initialize_graph()
        self.draw_network_interval = draw_network_interval

    @property
    def nodes(self):
        return np.array(self.graph.nodes(data=True))

    def set_agent_by_id(self, id: int, new_agent: agents.Agent):
        assert id == new_agent.id, "New agent must have the given ID as an attribute"
        for i, agent in enumerate(self.agent_list):
            if agent.id == id:
                self.agent_list[i] = new_agent
                self.graph.nodes[i]['data'] = new_agent
                return
        raise ValueError(f"Can't update agent: ID {id} not found in agent list.")

    def initialize_graph(self):
        graph = self.graph_type(self.n_agents, self.m_edges_per_node)
        for i, agent in enumerate(self.agent_list):
            graph.nodes[i]['data'] = agent
        return graph

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

    def node_social_learning(self, learner_agent, neighbor_agent, a: float, b: float) -> None:

        p_switch = 1/(1+np.exp(a+b*(learner_agent.payoff - neighbor_agent.payoff)))

        print("p_switch", p_switch)
        r = np.random.uniform()

        if r < p_switch:

            print(f"Switch between: Learner {learner_agent} with {learner_agent.payoff}"
                  f" and Neighbor {neighbor_agent} with {neighbor_agent.payoff} payoff")
            print(f"old learner: {learner_agent}")
            print(f"old neighbor: {neighbor_agent}")
            # previous_payoff = learner_agent.payoff
            # previous_id = learner_agent.id
            # learner_agent = deepcopy(neighbor_agent)
            # learner_agent.payoff = previous_payoff
            # learner_agent.id = previous_id
            learner_agent = learner_agent.transmute(neighbor_agent)
            print(f"new learner: {learner_agent}")
            print(f"new neighbor: {neighbor_agent}")

            self.set_agent_by_id(learner_agent.id, learner_agent)
            print(self.agent_list)

    def play_game_process(self):
        node1, node2 = my_graph.get_random_edge()
        agent1 = my_graph.graph.nodes[node1]['data']
        agent2 = my_graph.graph.nodes[node2]['data']
        my_graph.play_game(agent1, agent2)

    def social_learning_process(self):

        learner_node = my_graph.get_random_node()
        neighbor_node = my_graph.get_random_neighbor(learner_node)

        learner_agent = my_graph.graph.nodes[learner_node]['data']
        neighbor_agent = my_graph.graph.nodes[neighbor_node]['data']

        if neighbor_agent.type != learner_agent.type:
            my_graph.node_social_learning(learner_agent, neighbor_agent, a=0.5, b=0.5)

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
        plt.title(f"Barabási–Albert Network (generation {generation})")
        plt.show()

    def iterate_generations(self, p_play_game: float, p_social_learning: float, plot=False):
        for i_gen in range(self.generations):
            if random.random() < p_play_game:
                self.play_game_process()
            if random.random() < p_social_learning:
                self.social_learning_process()
            # Todo: rewiring

            self.store_agent_counts()
            if i_gen % self.draw_network_interval == 0:
                self.draw_network(i_gen)
        if plot:
            self.plot_agent_counts()


    def plot_agent_counts(self):
        for agent_type, counts in self.agent_counts.items():
            plt.plot(np.array(counts) / self.n_agents, label=agent_type.__name__)
        plt.title('Plot of agent populations over generations')
        plt.xlabel('Generations')
        plt.ylabel('Population')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()


if __name__ == '__main__':
    my_graph = SatisfiaMaximiserNetwork(
        JOBST_GAME,
        combined_strategies,
        10,
        0.9,
        100,
        2,
        nx.barabasi_albert_graph,
        10
    )
    my_graph.iterate_generations(1, 0.5, True)
    #
    # PLAY_GAME_PROB = 1
    # LEARN_PROB = 1
    # my_graph.draw_network()
    #
    # for i in range(my_graph.generations):
    #     play_game_prob, learn_prob = np.random.uniform(size=2)
    #
    #     if play_game_prob < PLAY_GAME_PROB:
    #         my_graph.play_game_process()
    #
    #     if learn_prob < LEARN_PROB:
    #         my_graph.social_learning_process()
    #         print(my_graph.agent_counts)
    #





