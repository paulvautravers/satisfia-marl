import monte_carlo as mc
import game
import agents

satisfia_set = agents.satisfia_set

def iterate_over_n0(max_n0):

    for i in range(max_n0):

        satisfias = [agents.SatisfiaAgent(strategy_set=satisfia_set) for _ in range(max_n0 - i)]
        maximisers = [agents.SatisfiaAgent(strategy_set=maximisers) for _ in range(i)]


