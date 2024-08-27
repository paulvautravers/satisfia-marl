import unittest
import numpy as np
from abc import ABC

from agents import Agent


class TesterAgent(Agent):

    color = "blue"

    def get_action(self, opponent: Agent) -> int: return 0  # basic implementation for testing


class TesterAgent2(Agent):

    color = "red"

    def get_action(self, opponent: Agent) -> int: return 1

class AgentTests(unittest.TestCase):

    def setUp(self):
        Agent.reset()

    def test_agent_initialisation(self):
        """Test that properties of agents are correctly initialised"""
        agent = TesterAgent()
        self.assertEqual(agent.id, 0)
        self.assertEqual(agent.payoff, 0)
        self.assertEqual(agent.gamma, 1)

    def test_agent_count_increment(self):
        """Test that incrementing of agent count is correct"""
        num_tester_agents_to_check = 5

        agents = [TesterAgent() for _ in range(num_tester_agents_to_check)]
        for i, agent in enumerate(agents):
            with self.subTest(msg=f"Agent ID = {i}"):
                self.assertEqual(i, agent.id)

        with self.subTest(msg=f"Total agent count of Tester class = {num_tester_agents_to_check}"):
            self.assertEqual(TesterAgent.agent_count, num_tester_agents_to_check)

        tester_agent_2 = TesterAgent2()
        with self.subTest(msg=f"Total agent count of Agent class = {num_tester_agents_to_check+1}"):
            self.assertEqual(Agent.agent_count, num_tester_agents_to_check+1)

    def test_reset_agent_count(self):

        agent1 = TesterAgent()
        TesterAgent.reset()
        agent2 = TesterAgent()

        self.assertEqual(agent2.id, 0)
        self.assertEqual(TesterAgent.agent_count, 1)

    def test_transmute_deep_copy(self):

        agent1 = TesterAgent()
        agent2 = TesterAgent2()
        agent2.payoff = 1

        agent_copy = agent1.transmute(agent2)

        with self.subTest(msg="ID and payoffs preserved"):
            self.assertEqual(agent_copy.id, agent1.id)
            self.assertEqual(agent_copy.payoff, agent1.payoff)

        with self.subTest(msg="Agent type converted"):
            self.assertEqual(agent_copy.__class__, TesterAgent2)

    def test_correct_avg_payoff(self):
        agent = TesterAgent()
        agent.payoff = 5
        gammas = [0, 0.5, 1]
        new_payoffs = [-1, 0, 1]
        for p in new_payoffs:
            for g in gammas:
                with self.subTest(f"Checking avg payoff: new payoff {p}, gamma {g}"):
                    agent.gamma = g
                    new_payoff = agent.get_new_avg_payoff(p)
                    expected_payoff = agent.gamma*agent.payoff + p
                    self.assertEqual(new_payoff,expected_payoff)

    def test_get_action_implemented(self):
        agent = TesterAgent()
        opponent = TesterAgent2()
        action = agent.get_action(opponent)
        self.assertEqual(action,0)

    def test_type_property(self):
        agent = TesterAgent()
        self.assertEqual(agent.type, TesterAgent)

    def test_label_property(self):
        agent = TesterAgent()
        self.assertEqual(agent.label,"TesterAgent")

    def test_color_property(self):
        agent = TesterAgent()
        self.assertEqual(agent.color,"blue")

    def test_agent_is_abstract(self):
        with self.assertRaises(TypeError):
            agent = Agent()

