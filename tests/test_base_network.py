import unittest
import numpy as np
from network.base_class import gen_agent_population, SatisfiaAgent, MaximiserAgent


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

