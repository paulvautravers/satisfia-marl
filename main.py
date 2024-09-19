import matplotlib.pyplot as plt
import numpy as np

import games
from agents import Agent, SatisfiaAgent, MaximiserAgent, SATISFIA_SET, MAXIMISER_SET
from games import JOBST_GAME
from monte_carlo import MonteCarlo as _mc, combined_strategies


def iterate_over_n0(max_n0: int, game: games.Game, combined_strategies: dict,
                    generations: int = 10):

    counts = []
    for i in range(max_n0):

        satisfias = [SatisfiaAgent(strategy_set=SATISFIA_SET) for _ in range(max_n0 - i)]
        maximisers = [MaximiserAgent(strategy_set=MAXIMISER_SET) for _ in range(i)]
        population = np.append(satisfias, maximisers)

        mc = _mc(game, combined_strategies, population, generations)
        count = mc.iterate_generations(plot=False)
        counts.append(count)
        Agent.reset()
    return counts

def legend_without_duplicate_labels(axis):
    handles, labels = axis.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


n_agents = 100
counts = iterate_over_n0(n_agents, JOBST_GAME, combined_strategies,
                generations=25)

fig, ax = plt.subplots()
for i, count in enumerate(counts):
    for agent_type, agent_counts in count.items():
        ax.plot(np.array(agent_counts) / n_agents,
                label=agent_type.__name__ if i == 1 else None,
                c=agent_type.color, alpha=0.3)

    ax.set_title('Plot of agent populations over generations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Agent type share of population')
    ax.set_ylim([0, 1])
    ax.legend()

# legend_without_duplicate_labels(ax)
plt.show()
