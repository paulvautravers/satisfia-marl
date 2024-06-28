import numpy as np
import numpy.typing as npt

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


jobst_game = np.array([-2, -2, -1, -3, 0, -4, 1, -4, 3, -4, 7, -1,
                       -3, -1, 0, 0, 0, -1, 1, -2, 3, -2, 6, -4,
                       -4, 0, -1, 0, -1, -1, 1, 0, 4, 1, 6, -1,
                       -4, 1, -2, 1, 0, 1, 2, 2, 3, 1, 5, 1,
                       -4, 3, -2, 3, 1, 4, 1, 3, 3, 3, 5, 3,
                       -1, 7, -4, 6, -1, 6, 1, 5, 3, 5, 5, 5])

jobst_game = jobst_game.reshape(6, 6, 2)
JobstGame = Game(jobst_game)


if __name__ == '__main__':
    prisoners_dilemma = np.array([[(1, 1), (3, 0)], [(0, 3), (2, 2)]], dtype=(int, int))
    # prisoners_dilemma = prisoners_dilemma.reshape((2, 2))
    defaultGame = Game(prisoners_dilemma)

    JobstGame = Game(jobst_game)
    payoff = JobstGame.get_reward(3, 1)
    print(payoff)

    print(JobstGame.row_options)

