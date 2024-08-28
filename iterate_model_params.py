from network import SatisfiaMaximiserNetwork
import argparse as ap
from games import JOBST_GAME
from monte_carlo import combined_strategies

import networkx as nx

GAME = JOBST_GAME
GENERATIONS = 1000
REPEATS = 20
def iterate_params_argparse():

    parser = ap.ArgumentParser()
    # Fundamental arguments for Network Constructor
    parser.add_argument("-gT", "--graph_type",
                        help="Type of graph to use as base for network",
                        choices=["barabasi", "fully-connected", "strogatz"], required=True)
    parser.add_argument("-nA", "--number_agents",
                        help="Total number of agents in the population (equal to # nodes)", required=True)
    parser.add_argument("-sS", "--satisfia_share",
                        help="Share of initial total agent population that are Satisfia", required=True)
    # Network Process params
    parser.add_argument("-gP", "--game_probability",
                        help="Probability of playing a game on a given edge")
    parser.add_argument("-lP", "--learn_probability",
                        help="Probability of learning for a given node")
    parser.add_argument("-lA", "--learn_param_a",
                        help="Value for param A of learning process")
    parser.add_argument("-lB", "--learn_param_b",
                        help="Value for param B of learning process")
    # Agent Params
    # Todo: we don't use this or allow for it in the constructor yet
    parser.add_argument("-aG", "--agent_gamma",
                        help="Gamma value for agents in network")

    # Graph specific args
    parser.add_argument("-eB", "--edges_barabasi",
                        help="Number of edges per node")
    parser.add_argument("-nS", "--neighbors_strogatz",
                        help="Neighbours for each node in Watts-Strogatz")
    parser.add_argument("-pS", "--probability_strogatz",
                        help="Probability of rewiring in Watts-Strogatz")

    parser.add_argument()
    args = parser.parse_args()

    return vars(args)

if __name__ == "__main__":

    args = iterate_params_argparse()

    graph_type = args['graph_type']
    n_agents = args['number_agents']
    satisfia_share = args['satisfia_share']

    match graph_type:
        case 'barabasi':
            graph = nx.barabasi_albert_graph(n_agents, args['edges_barabasi'])
        case 'strogatz':
            graph = nx.watts_strogatz_graph(n_agents, args['neighbors_strogatz'], args['probability_strogatz'])
        case 'fully-connected':
            graph = nx.complete_graph(n_agents)
        case '_':
            raise AssertionError("Graph type must have a value")

    network_instance = SatisfiaMaximiserNetwork(game=JOBST_GAME,
                                                strategy_dict=combined_strategies,
                                                satisfia_share=satisfia_share,
                                                generations=GENERATIONS,
                                                base_graph=graph,
                                                draw_network_interval=10000,  # no drawing of networks
                                                learn_param_a=args['learn_param_a'],
                                                learn_param_b=args['learn_param_b']
                                                )
