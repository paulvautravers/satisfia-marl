import unittest
from unittest.mock import MagicMock
from mock import Mock

import numpy as np
from network.base_class import gen_agent_population, SatisfiaMaximiserNetwork
from agents import SatisfiaAgent, MaximiserAgent, SATISFIA_SET, MAXIMISER_SET
import networkx as nx
from math import factorial

from games import JOBST_GAME

class TestGenAgentPopulation(unittest.TestCase):

    def test_basic_functionality(self):
        n_agents = 10
        satisfia_share = 0.5
        population = gen_agent_population(n_agents, satisfia_share)
        self.assertEqual(len(population), n_agents)
        self.assertEqual(sum(isinstance(agent, SatisfiaAgent) for agent in population), 5)
        self.assertEqual(sum(isinstance(agent, MaximiserAgent) for agent in population), 5)

    def test_zero_agents(self):
        n_agents = 0
        satisfia_share = 0.5
        population = gen_agent_population(n_agents, satisfia_share)
        self.assertEqual(len(population), 0)

    def test_rounding_behavior(self):

        n_agents = 10
        satisfia_shares = [0.0, 0.23, 0.49, 0.51, 0.58, 0.91, 1.0]
        for share in satisfia_shares:
            expected_satisfia_count = round(share*n_agents)
            with self.subTest(msg=f"With share {share}, Satisfia Count should be: {expected_satisfia_count}"):
                population = gen_agent_population(n_agents, share)
                self.assertEqual(sum(isinstance(agent, SatisfiaAgent) for agent in population), expected_satisfia_count)

class TestSatisfiaMaximiserNetwork(unittest.TestCase):

    N_AGENTS = 20
    N_GENERATIONS = 50
    SATISFIA_SHARE = 0.5

    def setUp(self):
        self.graph = nx.complete_graph(self.N_AGENTS)  # Create a simple complete graph with 5 nodes
        self.game = JOBST_GAME  # Substitute with a proper Game class instance
        self.strategy_dict = {}  # Substitute with appropriate strategies
        self.network = SatisfiaMaximiserNetwork(
            game=self.game,
            strategy_dict=self.strategy_dict,
            satisfia_share=self.SATISFIA_SHARE,
            generations=self.N_GENERATIONS,
            base_graph=self.graph,
            draw_network_interval=2
        )

    def test_initialization(self):
        self.assertEqual(len(self.network.agent_list), self.N_AGENTS)
        satisfia_count = sum(isinstance(agent, SatisfiaAgent) for agent in self.network.agent_list)
        self.assertEqual(satisfia_count, round(self.N_AGENTS*self.SATISFIA_SHARE))

    def test_reset(self):
        iterations = 1000
        agent_list_equal_count = 0
        n_agents = len(self.network.agent_list)
        for i in range(iterations):
            initial_agents = list(self.network.agent_list)
            self.network.reset()
            new_agents = list(self.network.agent_list)
            if initial_agents == new_agents:
                agent_list_equal_count += 1

        self.assertAlmostEqual(agent_list_equal_count/iterations, 1.0/factorial(self.N_AGENTS))

    def test_set_agent_by_id(self):
        agent = self.network.agent_list[0]
        new_agent = SatisfiaAgent(SATISFIA_SET)
        id_to_set = agent.id
        new_agent.id = id_to_set
        self.network.set_agent_by_id(id_to_set, new_agent)
        self.assertEqual(self.network.agent_list[0], new_agent)

    def test_set_agent_by_invalid_id(self):
        id = 0
        with self.assertRaises(AssertionError):
            self.network.set_agent_by_id(id, MaximiserAgent(MAXIMISER_SET))

    def test_set_invalid_agent_by_id(self):

        id = 99
        new_agent = SatisfiaAgent(SATISFIA_SET)
        new_agent.id = id

        with self.assertRaises(ValueError):
            self.network.set_agent_by_id(id, new_agent)

    def test_node_social_learning(self):
        #Todo ... This test needs to be developed; potentially with mock objects at first?
        pass

    def test_play_game_process_function_calls(self):
        #Todo ... This test needs to be elaborated beyond just mock call counts
        self.network.play_game = Mock()
        self.network.get_random_edge = Mock()

        mock_node_1, mock_node_2 = Mock(), Mock()
        self.network.get_random_edge.return_value = mock_node_1, mock_node_2

        self.network.graph.nodes = MagicMock()
        mock_dict = {mock_node_1: {'data': Mock()}, mock_node_2: {'data': Mock()}}
        self.network.graph.nodes.__getitem__.side_effect = mock_dict.__getitem__

        self.network.play_game_process()

        self.assertEqual(self.network.play_game.call_count, 1)
        self.assertEqual(self.network.get_random_edge.call_count, 1)

    def test_iterate_generations(self):
        #Todo parameterised game, learning rates?
        self.network.play_game_process = Mock()
        self.network.social_learning_process = Mock()
        self.network.store_agent_counts = Mock()
        self.network.store_avg_closeness_centrality = Mock()

        n_iterations = 5000
        base_exp_counts = n_iterations*(self.N_GENERATIONS - 1)

        exp_game_rate, exp_learning_rate = 0.5, 0.5
        call_counts = {'agent_store': 0, 'centrality_store': 0,
                       'game': 0, 'learning': 0}

        for _ in range(n_iterations):
            __ = self.network.iterate_generations(exp_game_rate, exp_learning_rate)

        call_counts['agent_store'] += self.network.store_agent_counts.call_count
        call_counts['centrality_store'] += self.network.store_avg_closeness_centrality.call_count
        call_counts['game'] += self.network.play_game_process.call_count
        call_counts['learning'] += self.network.social_learning_process.call_count

        self.assertEqual(call_counts['agent_store'], base_exp_counts)
        self.assertEqual(call_counts['centrality_store'], base_exp_counts)

        self.assertAlmostEqual(call_counts['game']/base_exp_counts, exp_game_rate, places=2)
        self.assertAlmostEqual(call_counts['learning']/base_exp_counts, exp_game_rate, places=2)

    def test_get_avg_closeness_centrality(self):
        centrality = self.network.get_avg_closeness_centrality()
        self.assertIsInstance(centrality, dict)
        centrality_satisfia = self.network.get_avg_closeness_centrality(SatisfiaAgent)
        self.assertIsInstance(centrality_satisfia, float)

    def test_get_avg_degree(self):
        avg_degree = self.network.get_avg_degree()
        self.assertIsInstance(avg_degree, float)

if __name__ == '__main__':
    unittest.main()