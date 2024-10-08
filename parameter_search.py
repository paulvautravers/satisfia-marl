import os
import csv
import argparse as ap
import numpy as np

import random
from networkx import barabasi_albert_graph, watts_strogatz_graph, complete_graph

import games
from network import SatisfiaMaximiserNetwork, NetworkByNeighborhood, NetworkByCentrality
from games import JOBST_GAME
from monte_carlo import combined_strategies
from agents import SatisfiaAgent

from typing import Optional, List, Union

GAME = JOBST_GAME


class ParameterSearch:

    def __init__(self, generations: int, repeats: int, simulations_per_task: int, output_dir: Optional[str] = '',
                 game: games.Game = JOBST_GAME, strategy_dict: dict = combined_strategies):
        """"
        Fixed Inputs -> Not explored in search
        """
        self.generations = generations
        self.repeats = repeats
        self.simulations_per_task = simulations_per_task
        self.game = game
        self.strategy_dict = strategy_dict
        """
        Random variables -> Explored in search
        """
        self.n_agents, self.satisfia_share = None, None
        self.game_probability, self.learn_probability = None, None
        self.learn_param_a, self.learn_param_b = None, None  # Todo Check what range this can have

        self.network_constructor, self.centrality_shift, self.graph_type, self.graph = None, None, None, None
        self.edges_barabasi, self.neighbours_strogatz, self.strogatz_probability = None, None, None

        self.satisfia_trajectory = None
        """
        Slurm environmental variables -> Random seeds
        """
        self.slurm_pid = int(os.environ.get('SLURM_PROCID', 0))
        self.slurm_job_id = int(os.environ.get('SLURM_JOB_ID', 0))
        self.unique_sim_seed = self.slurm_pid*self.simulations_per_task

        self.output_filename = f"slurm_param_search_job_{self.slurm_job_id}_pid_{self.slurm_pid}.csv"
        self.output_filepath = os.path.join(output_dir, self.output_filename)
        self.create_file(self.full_output_dict)

    @property
    def full_output_dict(self) -> dict:
        return {'slurm_pid': self.slurm_pid, 'unique_sim_seed': self.unique_sim_seed,
                'n_agents': self.n_agents, 'satisfia_share': self.satisfia_share,
                'game_probability': self.game_probability, 'learn_probability': self.learn_probability,
                'learn_param_a': self.learn_param_a, 'learn_param_b': self.learn_param_b,
                'network_constructor': self.network_constructor, 'shift_most_central': self.centrality_shift,
                'base_graph': self.graph_type, 'edges_barabasi': self.edges_barabasi,
                'neighbours_strogatz': self.neighbours_strogatz, 'strogatz_probability': self.strogatz_probability,
                'satisfia_trajectory': self.satisfia_trajectory}

    @property
    def network_param_dict(self) -> dict:

        network_param_dict = {'game': self.game, 'strategy_dict': self.strategy_dict, 'satisfia_share': self.satisfia_share,
                              'generations': self.generations, 'base_graph': self.graph, 'draw_network_interval': 1e9,
                              'learn_param_a': self.learn_param_a, 'learn_param_b': self.learn_param_b}

        if self.network_constructor == NetworkByCentrality:
            network_param_dict['shift_most_central'] = self.centrality_shift

        return network_param_dict

    def zero_simulation_attrs(self):
        self.n_agents, self.satisfia_share = None, None
        self.game_probability, self.learn_probability = None, None
        self.learn_param_a, self.learn_param_b = None, None  # Todo Check what range this can have

        self.network_constructor, self.centrality_shift, self.graph_type, self.graph = None, None, None, None
        self.edges_barabasi, self.neighbours_strogatz, self.strogatz_probability = None, None, None

        self.satisfia_trajectory = None

    def set_random_params(self):
        """
        Randomizes relevant parameters being searched over
        """
        self.n_agents = random.randint(10, 500)*2
        self.satisfia_share = random.random()

        self.game_probability = random.random()
        self.learn_probability = random.random()
        self.learn_param_a = random.random()  # Todo Check what range this can have
        self.learn_param_b = random.random()  # Todo Check what range this can have

        self.network_constructor = random.choice([SatisfiaMaximiserNetwork, NetworkByNeighborhood, NetworkByCentrality])

        if self.network_constructor == NetworkByCentrality:
            self.centrality_shift = random.randint(0, self.n_agents-1)

        self.graph_type = random.choice(['barabasi_albert_graph', 'watts_strogatz_graph', 'complete_graph'])

        match self.graph_type:
            case 'barabasi_albert_graph':
                self.edges_barabasi = random.randint(1, 6)
                self.graph = barabasi_albert_graph(self.n_agents, self.edges_barabasi)
            case 'watts_strogatz_graph':
                self.neighbours_strogatz = random.randint(1, 3)*2  # K neighbours must be even for strogatz!
                self.strogatz_probability = random.random()
                self.graph = watts_strogatz_graph(self.n_agents, self.neighbours_strogatz, self.strogatz_probability)
            case 'complete_graph':
                self.graph = complete_graph(self.n_agents)

    def get_random_network(self) -> Optional[Union[SatisfiaMaximiserNetwork, NetworkByNeighborhood, NetworkByCentrality]]:
        return self.network_constructor(**self.network_param_dict)

    def create_file(self, dict_to_write: dict):
        with open(self.output_filepath, "w+", newline='') as f:
            writer = csv.DictWriter(f, dict_to_write.keys())
            writer.writeheader()

    def write_to_file(self, dict_to_write: dict):
        with open(self.output_filename, "a", newline='') as f:
            writer = csv.DictWriter(f, dict_to_write.keys())
            writer.writerow(dict_to_write)

    def run_simulations(self):
        for i in range(self.simulations_per_task):

            self.unique_sim_seed += 1
            random.seed(self.unique_sim_seed)  # Ensure that params for each simulation run are reproducible

            self.set_random_params()
            network_simulator = self.get_random_network()
            print(self.full_output_dict)

            trajectories = network_simulator.get_iteration_repeats(self.game_probability, self.learn_probability,
                                                                   self.repeats)
            self.satisfia_trajectory = np.mean(trajectories[SatisfiaAgent], axis=0)

            output_dict = self.full_output_dict
            self.write_to_file(output_dict)

            self.zero_simulation_attrs()
            print("\n")



def param_search_argparse():

    parser = ap.ArgumentParser()
    # Fundamental arguments for parameter search
    parser.add_argument("-g", "--generations", default=10, help="Number of generations to track populations")
    parser.add_argument("-r", "--repeats", default=1, help="Number of repeats for each trajectory")
    parser.add_argument("-s", "--simulations_per_task", default=10,
                        help="Number of simulations to be ran consecutively, under one task")
    return vars(parser.parse_args())


if __name__ == "__main__":

    args = param_search_argparse()

    pS = ParameterSearch(int(args['generations']), int(args['repeats']), int(args['simulations_per_task']),
                         game=JOBST_GAME, strategy_dict=combined_strategies)

    pS.run_simulations()


