import numpy as np


class Agent:

    agent_dict = {}  # initialise to zero

    def __init__(self, label: str, strategy_set: dict = None, opponent=None, strategy=None):

        self.label = label
        self.strategy_set = strategy_set
        self.strategy = strategy
        self.opponent = opponent
        self.id = Agent.agent_dict[label] + 1 if label in Agent.agent_dict else 0   ## I don't understand this
        Agent.agent_dict[label] = self.id

        print(Agent.agent_dict)

    def __repr__(self):
        return (f"Label: {self.label} \n"
                f"ID: {self.id} \n"
                f"Strategy Set: {self.strategy_set} \n")

    def __copy__(self):
        new_agent = Agent(self.label, self.strategy_set)
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

    def __init__(self, strategy_set):
        super().__init__('satisfia', strategy_set)

class MaximiserAgent(Agent):

    def __init__(self, strategy_set):
        super().__init__('maximiser', strategy_set)


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

    satisfier = Agent('satisfia')
    satisfier2 = Agent('satisfia')
    satisfier3 = Agent('satisfia')

    maximiser = Agent('maximiser')
    maximiser2 = Agent('maximiser')
    maximiser3 = maximiser.__copy__()













