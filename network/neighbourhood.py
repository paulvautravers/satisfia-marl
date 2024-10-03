import random

import networkx as nx
import numpy as np

from agents import SatisfiaAgent, MaximiserAgent, SATISFIA_SET, MAXIMISER_SET
from games import Game, JOBST_GAME
from network import SatisfiaMaximiserNetwork
from collections import deque


class NetworkByNeighborhood(SatisfiaMaximiserNetwork):
    def __init__(self,
                 game: Game,
                 strategy_dict: dict,
                 satisfia_share: float,
                 generations: int,
                 base_graph: nx.Graph,
                 draw_network_interval: int,
                 learn_param_a: float = 0.5,
                 learn_param_b: float = 0.5,
                 ):
        super().__init__(game, strategy_dict, satisfia_share,
                         generations, base_graph, draw_network_interval, learn_param_a, learn_param_b)

    def _initialize_graph(self, base_graph: nx.Graph, agent_list_to_ignore):
        graph = base_graph.copy()
        n_satisfia_to_assign = int(len(graph) * self.satisfia_share)
        queue = deque()
        assigned = []
        starting_node_idx = np.random.randint(0, len(graph))
        queue.append(starting_node_idx)

        while n_satisfia_to_assign > 0:
            # Get a new node
            current_node_idx = queue.popleft()

            # Assign it as SatisfIA
            new_agent = SatisfiaAgent(SATISFIA_SET)
            graph.nodes[current_node_idx]['data'] = new_agent
            assigned.append(current_node_idx)
            n_satisfia_to_assign -= 1

            # Store its neighbors in the queue
            neighbors = list(graph.neighbors(current_node_idx))
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in assigned:
                    queue.append(neighbor)

        # Assign all remaining nodes as Maximizer
        for node_idx in range(len(graph)):
            if 'data' not in graph.nodes[node_idx]:
                default_agent = MaximiserAgent(MAXIMISER_SET)
                graph.nodes[node_idx]['data'] = default_agent

        return graph



if __name__ == '__main__':
    N_AGENTS = 50
    EDGES_PER_NODE = 2
    BASE_BARABASI = nx.barabasi_albert_graph(N_AGENTS, EDGES_PER_NODE)

    BASE_FULL = nx.complete_graph(N_AGENTS)
    from monte_carlo import combined_strategies

    my_graph = NetworkByNeighborhood(
        JOBST_GAME,
        combined_strategies,
        0.4,
        60,
        BASE_BARABASI,
        50
    )
    my_graph.draw_network(0)
    trajectory = my_graph.iterate_generations(1, 1, plot=True)
    print(trajectory)
