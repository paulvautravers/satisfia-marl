from agents import Agent
from games import Game
import numpy as np


class TesterAgent(Agent):

    @property
    @classmethod
    def color(cls): return "blue"

    def get_action(self, opponent: Agent) -> int: return 0  # basic implementation for testing


class TesterAgent2(Agent):
    @property
    @classmethod
    def color(cls): return "red"

    def get_action(self, opponent: Agent) -> int: return 1

prisoners_dilemma = np.array([[(1, 1), (3, 0)], [(0, 3), (2, 2)]], dtype=(int, int))
defaultGame = Game(prisoners_dilemma)