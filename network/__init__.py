import networkx as netx
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt


import agents
from agents import SatisfiaAgent, MaximiserAgent, SATISFIA_SET, MAXIMISER_SET
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
        self.edges = self.graph.edges

    def set_node_types(self):
        graph = self.graph_type(self.n, self.m)
        r_vals = np.random.uniform(0, 1, size=self.n)
        for node, i in enumerate(graph.nodes()):
            # if r_vals[i] < self.satisfia_share:
            #     graph.nodes[node]['data'] = self.agent_list
            #     self.color_map.append('blue')
            # else:
            #     graph.nodes[node]['data'] = MaximiserAgent(MAXIMISER_SET)
            #     self.color_map.append('red')
            agent = self.agent_list[i]
            graph.nodes[node]['data'] = agent
            self.color_map.append(self.color_agent_mapping[agent.type])

        return graph

    def random_process_on_edge(self, edge_process_prob: float):
        r = np.random.uniform()
        if r < edge_process_prob:
            edge_idx = np.random.randint(self.edges)
            chosen_edge = self.edges[edge_idx]

            return chosen_edge

    def draw_network(self):
        plt.figure(figsize=(7, 7))
        nx.draw(self.graph, node_color=self.color_map, node_size=30, with_labels=False)
        plt.title("Barabási–Albert Network")
        plt.show()


if __name__ == '__main__':
    my_graph = SatisfiaMaximiserNetwork(
        JOBST_GAME,
        combined_strategies,
        10,
        0.5,
        100,
        3,
        nx.barabasi_albert_graph
    )

    print(my_graph.graph.nodes[1])

    for i in range(my_graph.n):
        print(my_graph.graph.nodes[i])

    my_graph.draw_network()





