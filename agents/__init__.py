from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Type

import numpy as np


class Agent(ABC):

    agent_dict = defaultdict(int)  # initialised to zero
    agent_count = 0

    def __init__(self):
        self.id = Agent.agent_count  ##  The id should be unique irrespective of the type
        Agent.agent_count += 1
        Agent.agent_dict[self.type] += 1

    def __repr__(self):
        return (f"Type: {self.type.__name__} \n"
                f"ID: {self.id} \n")

    def __str__(self):
        return f"{self.type.__name__}(ID={self.id})"

    @property
    def type(self) -> Type[Agent]:
        return self.__class__

    @property
    def label(self) -> str:
        return self.type.__name__

    @abstractmethod
    def get_action(self, opponent: Agent) -> int:
        raise NotImplementedError


class SatisfiaAgent(Agent):

    def __init__(self, strategy_set: dict):
        super().__init__()
        self.strategy_set = strategy_set

    def get_action(self, opponent: Agent) -> int:
        strategy = self.strategy_set[opponent.type]
        action = np.random.choice(strategy['actions'],
                                  p=strategy['probabilities'])
        return action


class MaximiserAgent(Agent):

    def __init__(self, strategy_set: dict):
        super().__init__()
        self.strategy_set = strategy_set

    def get_action(self, opponent: Agent) -> int:
        strategy = self.strategy_set[opponent.type]
        action = np.random.choice(strategy['actions'],
                                  p=strategy['probabilities'])
        return action


if __name__ == '__main__':

    # 'probabilities': (1/len(options)*np.ones_like(options))}

    options = [0, 1, 2, 3, 4, 5]
    satisfia_set = {'satisfia': {'actions': options,
                                 'probabilities': [0, 0, 0, 1, 0, 0]},
                    'maximiser': {'actions': options,
                                  'probabilities': [0, 1, 0, 0, 0, 0]}
                    }

    maximiser_set = {'satisfia': {'actions': options,
                                  'probabilities': [0, 1, 0, 0, 0, 0]},
                     'maximiser': {'actions': options,
                                   'probabilities': [0, 0, 0, 0, 0, 1]}
                     }

    satisfier = SatisfiaAgent(strategy_set=satisfia_set)
    maximiser = MaximiserAgent(strategy_set=maximiser_set)

    satisfier2 = SatisfiaAgent(strategy_set=satisfia_set)
    print(satisfier2.type)
    print(maximiser.type)
    print(repr(satisfier))
    print(maximiser)













