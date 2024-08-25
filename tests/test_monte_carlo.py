import unittest
import numpy as np
from monte_carlo import MonteCarlo
from utils import *
from collections import defaultdict

class MonteCarloTests(unittest.TestCase):

    N_AGENTS = 10  # per type

    def setUp(self):

        Agent.reset()
        self.game = defaultGame
        self.strategy_set = {TesterAgent: {TesterAgent: [0.5, 0.5], TesterAgent2: [0.5, 0.5]},  # 50-50 defect, cooperate
                             TesterAgent2: {TesterAgent: [0.5, 0.5], TesterAgent2: [0.5, 0.5]}}
        self.agents = [TesterAgent() for _ in range(self.N_AGENTS)] + [TesterAgent2() for _ in range(self.N_AGENTS)]
        self.agents = np.array(self.agents)
        self.generations = 50
        self.mc = MonteCarlo(self.game, self.strategy_set, self.agents, self.generations)

    def test_initialisation(self):
        self.assertEqual(self.mc.game, self.game)
        self.assertEqual(self.mc.strategy_set, self.strategy_set)
        self.assertEqual(self.mc.generations, self.generations)
        self.assertTrue(np.array_equal(self.mc.agent_list, self.agents))
        self.assertIsInstance(self.mc.reward_dict, defaultdict)

    def test_get_current_agent_coutnts(self):
        counts = self.mc.get_current_agent_counts()
        for i in range(self.N_AGENTS):
            new_agent = TesterAgent()
            self.mc.agent_list = np.append(self.mc.agent_list, new_agent)
            counts = self.mc.get_current_agent_counts()
            self.assertEqual(counts, {TesterAgent: self.N_AGENTS+(i+1), TesterAgent2: self.N_AGENTS})

    def test_store_agent_counts(self):

        self.mc.store_agent_counts()
        self.assertEqual(self.mc.agent_counts, {TesterAgent: [self.N_AGENTS, self.N_AGENTS],
                         TesterAgent2: [self.N_AGENTS, self.N_AGENTS]})

    def test_set_agent_by_id(self):  # this seems odd - is this desired behaviour?
        new_agent = TesterAgent()
        new_agent.id = 0
        self.mc.set_agent_by_id(0, new_agent)
        self.assertEqual(self.mc.agent_list[0], new_agent)

    def test_set_agent_with_invalid_id(self):
        new_agent = TesterAgent()
        new_agent.id = 999
        with self.assertRaises(ValueError):
            self.mc.set_agent_by_id(999, new_agent)

    def test_invalid_agent_by_id_set(self):
        new_agent = TesterAgent()
        new_agent.id = 37
        with self.assertRaises(AssertionError):
            self.mc.set_agent_by_id(5, new_agent)




