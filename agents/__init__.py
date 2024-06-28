import numpy as np
import copy
from game import Game, JobstGame

class Agent:

    agent_dict = {}  # initialise to zero

    def __init__(self, label: str, game: Game, strategy_set: dict = None,
                 opponent=None, strategy=None):

        self.label = label
        self.game = game
        self.strategy_set = strategy_set
        self.strategy = strategy
        self.opponent = opponent
        self.id = Agent.agent_dict[label] + 1 if label in Agent.agent_dict else 0
        Agent.agent_dict[label] = self.id

        print(Agent.agent_dict)

    def __repr__(self):
        return (f"Label: {self.label} \n"
                f"ID: {self.id} \n"
                f"Strategy Set: {self.strategy_set} \n")

    def __copy__(self):
        new_agent = Agent(self.label, self.game, self.strategy_set)
        return new_agent

    def set_opponent(self, agent):
        self.opponent = agent

    def set_strategy_set(self, strategy_set):
        self.strategy_set = strategy_set

    def choose_strategy(self):
        self.strategy = self.strategy_set[self.opponent.label]

    def get_action(self):
        action = np.random.choice(self.strategy['actions'],
                                  p=self.strategy['probabilities'])
        return action

class SatisfiaAgent(Agent):

    def __init__(self, game: Game, strategy_set):
        super().__init__('satisfia', game, strategy_set)

class MaximiserAgent(Agent):

    def __init__(self, game: Game, strategy_set):
        super().__init__('maximiser', game, strategy_set)


if __name__ == '__main__':

    # 'probabilities': (1/len(options)*np.ones_like(options))}

    options = JobstGame.row_options
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
    #
    # satisfier = SatisfiaAgent(JobstGame, strategy_set=satisfia_set)
    # maximiser = MaximiserAgent(JobstGame, strategy_set=maximiser_set)
    #
    # satisfier2 = SatisfiaAgent(JobstGame, strategy_set=satisfia_set)

    satisfier = Agent('satisfia', JobstGame)
    satisfier2 = Agent('satisfia', JobstGame)
    satisfier3 = Agent('satisfia', JobstGame)

    maximiser = Agent('maximiser', JobstGame)
    maximiser2 = Agent('maximiser', JobstGame)
    maximiser3 = maximiser.__copy__()













