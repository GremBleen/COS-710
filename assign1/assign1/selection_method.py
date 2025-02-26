from copy import deepcopy
from assign1.config_classes.config_manager import ConfigurationManager
from assign1.config_classes.component_factory import ComponentFactory
from assign1.population import Population
from assign1.utils import SingletonRandom
from assign1.fitness_functions import normalised_fitness_function

config_manager = ConfigurationManager()
component_factory = ComponentFactory()


def fitness_proportionate_selection(population=None):
    assert (
        config_manager.get_param("fitness").get("method") == "normalised"
    ), "Fitness method is not normalised fitness"

    if population is None:
        population = deepcopy(config_manager.get_config("population"))
    else:
        population = deepcopy(population)

    assert isinstance(
        population, Population
    ), "Population is not an instance of Population"

    randomboi = SingletonRandom()

    mating_pool = []

    pop_size = config_manager.get_param("population").get("size")

    for ind in population.individuals:
        num_copies = int(ind.fitness * pop_size)
        mating_pool.extend([ind] * num_copies)

    index = randomboi.randint(0, len(mating_pool) - 1)

    assert index < len(
        mating_pool
    ), "Index out of bounds, greater than mating pool size"

    assert index >= 0, "Index out of bounds, less than 0"

    return mating_pool[index]


def tournament_selection(population=None):

    if population is None:
        population = deepcopy(config_manager.get_config("population"))

    assert isinstance(
        population, Population
    ), "Population is not an instance of Population"

    randomboi = SingletonRandom()

    tournament_size = config_manager.get_param("selection").get("tournament_size")

    tournament = []

    for i in range(tournament_size):
        index = randomboi.randint(0, len(population.individuals) - 1)
        tournament.append(population.individuals[index])

    best = tournament[0]

    for ind in tournament:
        better_fitness = handle_better_fitness()

        if better_fitness == "low":
            if ind.fitness < best.fitness:
                best = ind
        elif better_fitness == "high":
            if ind.fitness > best.fitness:
                best = ind
        else:
            raise ValueError(f"Invalid better fitness: {better_fitness}")
        
    return best

def handle_better_fitness():
    method = config_manager.get_param("fitness").get("method")

    if method == "raw":
        return config_manager.get_param("fitness").get("better_fitness")
    elif method in ["standardised", "adjusted"]:
        return "low"
    elif method in ["normalised", "hit_rate"]:
        return "high"
    else:
        raise ValueError(f"Invalid fitness method: {method}")

# def rank_selection():
#     # TODO: Implement rank selection
#     raise NotImplementedError

# def linear_selection():
#     # TODO: Implement linear selection
#     raise NotImplementedError

# def exponential_selection():
#     # TODO: Implement exponential selection
#     raise NotImplementedError
