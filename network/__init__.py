import copy
import random

import networkx as netx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from copy import deepcopy


import agents
from agents import Agent, SatisfiaAgent, MaximiserAgent, SATISFIA_SET, MAXIMISER_SET
from monte_carlo import MonteCarlo, combined_strategies
from games import Game, JOBST_GAME


def gen_agent_population(n_agents: int, satisfia_share: float) -> npt.NDArray[agents.Agent]:
    satisfia_count, maximiser_count = round(n_agents*satisfia_share), round(n_agents*(1-satisfia_share))

    satisfias = np.array([agents.SatisfiaAgent(SATISFIA_SET) for _ in range(satisfia_count)])
    maximisers = np.array([agents.MaximiserAgent(MAXIMISER_SET) for _ in range(maximiser_count)])
    agent_population = np.append(satisfias, maximisers)
    np.random.shuffle(agent_population)
    return agent_population


class SatisfiaMaximiserNetwork(MonteCarlo):

    color_agent_mapping = {SatisfiaAgent: 'blue', MaximiserAgent: 'red'}

    def __init__(self, game: Game, strategy_dict: dict, n_agents: int, satisfia_share: float,
                 generations: int, edges_per_node: int, graph_type):

        self.n = n_agents
        self.satisfia_share = satisfia_share
        agent_list = gen_agent_population(self.n, self.satisfia_share)

        super().__init__(game, strategy_dict, agent_list, generations)
        self.m = edges_per_node
        self.satisfia_share = satisfia_share
        self.graph_type = graph_type
        self.color_map = []
        self.graph = self.set_node_types()
        self.edges = np.array(self.graph.edges)
        self.nodes = np.array(self.graph.nodes(data=True))

    def set_node_types(self):
        graph = self.graph_type(self.n, self.m)
        for node, i in enumerate(graph.nodes()):
            agent = self.agent_list[i]
            graph.nodes[node]['data'] = agent
            self.color_map.append(self.color_agent_mapping[agent.type])

        return graph

    def get_random_edge(self):
        edge_idx = np.random.randint(len(self.edges))
        chosen_edge = self.edges[edge_idx, :]
        return chosen_edge

    def get_random_node(self):
        node_idx = np.random.randint(self.n)
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
            previous_payoff = learner_agent.payoff
            previous_id = learner_agent.id
            learner_agent = deepcopy(neighbor_agent)
            learner_agent.payoff = previous_payoff
            learner_agent.id = previous_id
            print(f"new learner: {learner_agent}")
            print(f"new neighbor: {neighbor_agent}")


            self.agent_list[previous_id] = learner_agent
            print(self.agent_list)
            self.set_node_types()
            self.agent_counts = self.get_current_agent_counts()

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

    def draw_network(self):
        plt.figure(figsize=(7, 7))
        nx.draw(self.graph, node_color=self.color_map, node_size=30, with_labels=True)
        plt.title("Barabási–Albert Network")
        plt.show()


if __name__ == '__main__':
    my_graph = SatisfiaMaximiserNetwork(
        JOBST_GAME,
        combined_strategies,
        10,
        0.9,
        100,
        2,
        nx.barabasi_albert_graph
    )

    PLAY_GAME_PROB = 1
    LEARN_PROB = 1
    my_graph.draw_network()

    for i in range(my_graph.generations):
        play_game_prob, learn_prob = np.random.uniform(size=2)

        if play_game_prob < PLAY_GAME_PROB:
            my_graph.play_game_process()

        if learn_prob < LEARN_PROB:
            my_graph.social_learning_process()
            # my_graph.draw_network()
            print(my_graph.agent_counts)






