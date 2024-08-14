import networkx as nx
import numpy as np

from games import Game
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

    def initialize_graph(self, base_graph: nx.Graph):
        self.graph = base_graph  # Please review ## Maybe better to provide base graph to get_avg_closeness?
        nodes_by_closeness_centrality = self.get_avg_closeness_centrality()

        nodes_by_closeness_centrality = {k: v for k, v in
                                         sorted(nodes_by_closeness_centrality.items(), key=lambda item: item[1],
                                                reverse=True)}

        self.agent_list = np.roll(self.agent_list, self.shift_most_central)

        for i, node in enumerate(nodes_by_closeness_centrality.keys()):
            self.graph.nodes[node]['data'] = self.agent_list[i]

        return self.graph  ## Maybe better to provide base graph to get_avg_closeness?
