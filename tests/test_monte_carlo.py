import unittest
import numpy as np
from monte_carlo import MonteCarlo
from utils import *
from collections import defaultdict

class MonteCarloTests(unittest.TestCase):

    N_AGENTS = 10  # per type

    def setUp(self):
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
        self.assertEqual(self.mc.agent_counts, {TesterAgent: [self.N_AGENTS], TesterAgent2: [self.N_AGENTS]})
