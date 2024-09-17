import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from agents import SatisfiaAgent
from games import Game, JOBST_GAME
from monte_carlo import combined_strategies
from network import SatisfiaMaximiserNetwork


class NetworkByCentrality(SatisfiaMaximiserNetwork):

    def __init__(self,
                 game: Game,
                 strategy_dict: dict,
                 satisfia_share: float,
                 generations: int,
                 base_graph: nx.Graph,
                 draw_network_interval: int,
                 shift_most_central: int,
                 learn_param_a: float = 0.5,
                 learn_param_b: float = 0.5,
                 ):
        self.shift_most_central = shift_most_central
        super().__init__(game, strategy_dict, satisfia_share,
                         generations, base_graph, draw_network_interval)

    def _initialize_graph(self, base_graph: nx.Graph, agent_list):
        graph = base_graph.copy()
        nodes_by_closeness_centrality = nx.closeness_centrality(graph)

        nodes_by_closeness_centrality = {k: v for k, v in
                                         sorted(nodes_by_closeness_centrality.items(), key=lambda item: item[1],
                                                reverse=True)}

        agent_list = np.roll(agent_list, -self.shift_most_central)

        for i, node in enumerate(nodes_by_closeness_centrality.keys()):
            graph.nodes[node]['data'] = agent_list[i]

        return graph


if __name__ == '__main__':
    N_AGENTS = 30
    EDGES_PER_NODE = 2
    BASE_BARABASI = nx.barabasi_albert_graph(N_AGENTS, EDGES_PER_NODE)

    my_graph = NetworkByCentrality(
        JOBST_GAME,
        combined_strategies,
        0.4,
        400,
        BASE_BARABASI,
        200,
        9
    )
    my_graph.iterate_generations(1, 0.5, plot=True)
