import matplotlib.pyplot as plt
import numpy as np
import games
from monte_carlo import MonteCarlo as _mc, combined_strategies
from collections import defaultdict
from games import JOBST_GAME
from agents import Agent, SatisfiaAgent, MaximiserAgent, SATISFIA_SET, MAXIMISER_SET


def iterate_over_n0(max_n0: int, game: games.Game, combined_strategies: dict,
                    generations: int = 10, **kwargs):

    for i in range(max_n0):

        satisfias = [SatisfiaAgent(strategy_set=SATISFIA_SET) for _ in range(max_n0 - i)]
        maximisers = [MaximiserAgent(strategy_set=MAXIMISER_SET) for _ in range(i)]
        population = np.append(satisfias, maximisers)

        mc = _mc(game, combined_strategies, population, generations)
        mc.iterate_generations(**kwargs)

        Agent.reset()

def legend_without_duplicate_labels(axis):
    handles, labels = axis.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


fig, ax = plt.subplots()
iterate_over_n0(100, JOBST_GAME, combined_strategies,
                generations=25, plot=True, fig=fig, ax=ax, agent_to_plot=SatisfiaAgent)

legend_without_duplicate_labels(ax)
plt.show()
