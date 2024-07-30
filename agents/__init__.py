from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Type
import numpy as np

from globals import *


class Agent(ABC):

    agent_dict = defaultdict(int)  # initialised to zero
    agent_count = 0

    def __init__(self):
        self.id = Agent.agent_count  ##  The id should be unique irrespective of the type
        self.payoff = 0
        self.gamma = 1
        Agent.agent_count += 1
        # Agent.agent_dict[self.type] += 1

    def __repr__(self):
        return f"{self.type.__name__}(ID={self.id})"

    def __str__(self):
        return f"{self.type.__name__}(ID={self.id})"

    @classmethod
    def reset(cls):
        Agent.agent_dict = defaultdict(int)
        # Agent.agent_count = 0

    @property
    def type(self) -> Type[Agent]:
        return self.__class__

    @property
    def label(self) -> str:
        return self.type.__name__

    # @abstractmethod
    def get_action(self, opponent: Agent) -> int:
        # raise NotImplementedError
        pass

    def get_new_avg_payoff(self, new_payoff: float):
        return self.gamma*self.payoff + new_payoff

    # def __deepcopy__(self, memodict={}):
    #     copy_obj = copy.deepcopy(self)
    #     # copy_obj.payoff = 0
    #
    #     return copy_obj


class SatisfiaAgent(Agent):

    def __init__(self, strategy_set: dict):
        super().__init__()
        self.strategy_set = strategy_set

    def get_action(self, opponent: Agent) -> int:
        strategy = self.strategy_set[opponent.type]
        action = np.random.choice(range(len(strategy)), p=strategy)

        return action


class MaximiserAgent(Agent):

    def __init__(self, strategy_set: dict):
        super().__init__()
        self.strategy_set = strategy_set

    def get_action(self, opponent: Agent) -> int:
        strategy = self.strategy_set[opponent.type]
        action = np.random.choice(range(len(strategy)), p=strategy)
        return action


# Agent related globals
MAXIMISER_SET = {SatisfiaAgent: [0, 1, 0, 0, 0, 0], MaximiserAgent: [0, 0, 0, 0, 0, 1]}
SATISFIA_SET = {SatisfiaAgent: [0, 0, 0, 1, 0, 0], MaximiserAgent: [0, 1, 0, 0, 0, 0]}

if __name__ == '__main__':

    satisfia = SatisfiaAgent(strategy_set=SATISFIA_SET)
    maximiser = MaximiserAgent(strategy_set=MAXIMISER_SET)

    print(satisfia)
    print(maximiser)

    class_type = type(maximiser)
    new_agent = Agent() #class_type.__new__(class_type)
    new_agent.__class__ = class_type

    print(new_agent)

    # print(SATISFIA_SET)













