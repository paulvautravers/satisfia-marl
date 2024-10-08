import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from agents import Agent, MaximiserAgent, SatisfiaAgent, SATISFIA_SET, MAXIMISER_SET


class Game:

    def __init__(self, normal_form: npt.NDArray[tuple[int, int]]):
        self.normal_form = normal_form
        self.row_options = np.arange(len(self.normal_form[:, 0]))
        self.column_options = np.arange(len(self.normal_form[0, :]))

    def __repr__(self):
        return f"{self.normal_form}"

    def get_reward(self, row_option: int, column_option: int) -> tuple[int, int]:
        reward = self.normal_form[row_option, column_option]
        return reward

    def play_n_rounds(self, n: int, row_agent: Agent, column_agent: Agent, plot=False) -> tuple[npt.NDArray[int], npt.NDArray[int]]:
        actions = np.zeros((2, n), dtype=int)
        rewards = np.zeros((2, n), dtype=int)

        for round in range(n):
            row_action = row_agent.get_action(column_agent)
            column_action = column_agent.get_action(row_agent)
            row_reward, column_reward = self.get_reward(row_action, column_action)

            actions[:, round] = [row_action, column_action]
            rewards[:, round] = [row_reward, column_reward]

        if plot:
            plt.plot(rewards[0, :], label=f'{row_agent.label} (row agent)')
            plt.plot(rewards[1, :], label=f'{column_agent.label} (column agent)')
            plt.title("Rewards at every round")
            plt.xlabel('Round')
            plt.ylabel('Reward')
            plt.legend()
            plt.show()
        return actions, rewards


# Game related globals
jobst_game = np.array([-2, -2, -1, -3, 0, -4, 1, -4, 3, -4, 7, -1,
                       -3, -1, 0, 0, 0, -1, 1, -2, 3, -2, 6, -4,
                       -4, 0, -1, 0, -1, -1, 1, 0, 4, 1, 6, -1,
                       -4, 1, -2, 1, 0, 1, 2, 2, 3, 1, 5, 1,
                       -4, 3, -2, 3, 1, 4, 1, 3, 3, 3, 5, 3,
                       -1, 7, -4, 6, -1, 6, 1, 5, 3, 5, 5, 5])

jobst_game = jobst_game.reshape(6, 6, 2)
JOBST_GAME = Game(jobst_game)


if __name__ == '__main__':
    prisoners_dilemma = np.array([[(1, 1), (3, 0)], [(0, 3), (2, 2)]], dtype=(int, int))
    defaultGame = Game(prisoners_dilemma)

    # Iterated JobstGame example
    maximiser = MaximiserAgent(MAXIMISER_SET)
    satisfia = SatisfiaAgent(SATISFIA_SET)

    actions, rewards = JOBST_GAME.play_n_rounds(10, satisfia, maximiser, plot=True)

    print(actions)
    print(rewards)

