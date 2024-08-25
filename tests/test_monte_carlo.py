import unittest
from mock import Mock
import numpy as np
from monte_carlo import MonteCarlo
from utils import *
from collections import defaultdict


class MonteCarloTests(unittest.TestCase):

    N_AGENTS = 10  # per type
    N_GENERATIONS = 50

    def setUp(self):

        Agent.reset()
        self.game = defaultGame
        self.strategy_set = {TesterAgent: {TesterAgent: [0.5, 0.5], TesterAgent2: [0.5, 0.5]},  # 50-50 defect, cooperate
                             TesterAgent2: {TesterAgent: [0.5, 0.5], TesterAgent2: [0.5, 0.5]}}
        self.agents = [TesterAgent(self.strategy_set[TesterAgent]) for _ in range(self.N_AGENTS)] + [TesterAgent2(self.strategy_set[TesterAgent2]) for _ in range(self.N_AGENTS)]
        self.agents = np.array(self.agents)
        self.generations = self.N_GENERATIONS
        self.mc = MonteCarlo(self.game, self.strategy_set, self.agents, self.generations)

    def test_initialisation(self):
        self.assertEqual(self.mc.game, self.game)
        self.assertEqual(self.mc.strategy_set, self.strategy_set)
        self.assertEqual(self.mc.generations, self.generations)
        self.assertTrue(np.array_equal(self.mc.agent_list, self.agents))
        self.assertIsInstance(self.mc.reward_dict, defaultdict)

    def test_get_current_agent_counts(self):
        counts = self.mc.get_current_agent_counts()
        for i in range(self.N_AGENTS):
            new_agent = TesterAgent(self.strategy_set[TesterAgent])
            self.mc.agent_list = np.append(self.mc.agent_list, new_agent)
            counts = self.mc.get_current_agent_counts()
            self.assertEqual(counts, {TesterAgent: self.N_AGENTS+(i+1), TesterAgent2: self.N_AGENTS})

    def test_store_agent_counts(self):

        self.mc.store_agent_counts()
        self.assertEqual(self.mc.agent_counts, {TesterAgent: [self.N_AGENTS, self.N_AGENTS],
                         TesterAgent2: [self.N_AGENTS, self.N_AGENTS]})

    def test_set_agent_by_id(self):  # this seems odd - is this desired behaviour?
        new_agent = TesterAgent(self.strategy_set[TesterAgent])
        new_agent.id = 0
        self.mc.set_agent_by_id(0, new_agent)
        self.assertEqual(self.mc.agent_list[0], new_agent)

    def test_set_agent_with_invalid_id(self):
        new_agent = TesterAgent(self.strategy_set[TesterAgent])
        new_agent.id = 999
        with self.assertRaises(ValueError):
            self.mc.set_agent_by_id(999, new_agent)

    def test_invalid_agent_by_id_set(self):
        new_agent = TesterAgent(self.strategy_set[TesterAgent])
        new_agent.id = 37
        with self.assertRaises(AssertionError):
            self.mc.set_agent_by_id(5, new_agent)

    def test_play_game(self):
        # defaultGame = prisoners_dilemma = np.array([[(1, 1), (3, 0)], [(0, 3), (2, 2)]], dtype=(int, int))
        agent1 = TesterAgent(self.strategy_set[TesterAgent])  # maybe could use mocks here for the assert_called_with method?
        agent2 = TesterAgent2(self.strategy_set[TesterAgent2])

        self.mc.play_game(agent1, agent2)

        self.assertEqual(agent1.payoff, 3)
        self.assertEqual(agent2.payoff, 0)

    def test_play_all_games_call_count(self):
        self.mc.play_game = Mock()
        self.mc.play_all_games()
        # 2 * N_agents is number of total agents of both type
        self.assertEqual(self.mc.play_game.call_count, self.N_AGENTS)

    def test_set_new_population(self):
        proportions = [[0, 20], [5, 15], [10, 10], [15, 5], [20, 0]]

        for prop in proportions:
            with self.subTest():
                self.mc.reward_dict[TesterAgent] = prop[0]
                self.mc.reward_dict[TesterAgent2] = prop[1]
                self.mc.set_new_population()
                tester_agents_1 = [agent for agent in self.mc.agent_list if agent.type == TesterAgent]
                tester_agents_2 = [agent for agent in self.mc.agent_list if agent.type == TesterAgent2]

                self.assertEqual(len(tester_agents_1), prop[0])
                self.assertEqual(len(tester_agents_2), prop[1])

    def test_iterate_generations_call_counts(self):

        self.mc.shuffle_population = Mock()
        self.mc.play_all_games = Mock()
        self.mc.set_new_population = Mock()

        agent_counts = self.mc.iterate_generations(plot=False)

        self.assertEqual(self.mc.shuffle_population.call_count, self.N_GENERATIONS - 1)
        self.assertEqual(self.mc.play_all_games.call_count, self.N_GENERATIONS - 1)
        self.assertEqual(self.mc.set_new_population.call_count, self.N_GENERATIONS - 1)
        self.assertEqual(agent_counts, self.mc.agent_counts)









