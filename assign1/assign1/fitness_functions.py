import numpy as np
from copy import deepcopy
from assign1.config_classes.config_manager import ConfigurationManager

from assign1.syntax_tree import SyntaxTree

config_manager = ConfigurationManager()


def error_function(prediction: float, target: float):
    return np.sum(np.abs(prediction - target))


def raw_fitness_function(ind, predictions=None):
    assert isinstance(ind, SyntaxTree), "Individual is not an instance of SyntaxTree"

    if predictions is None:
        predictions = ind.predict()

    targets = config_manager.get_config("data").get("train_data").target

    error = []

    for i in range(len(predictions)):
        # print("PREDICTION: ", predictions[i])
        # print("TARGET: ", targets.iloc[i])
        # print("ERROR: ", error_function(predictions[i], targets.iloc[i]))
        error.append(error_function(predictions[i], targets.iloc[i]))

    ind.fitness = np.sum(error)

    return ind.fitness


def standardised_fitness_function(ind, predictions=None):
    assert isinstance(ind, SyntaxTree), "Individual is not an instance of SyntaxTree"

    better_fitness = config_manager.get_param("fitness").get("better_fitness")
    if better_fitness == "low":
        ind.fitness = raw_fitness_function(ind, predictions)
        return ind.fitness
    elif better_fitness == "high":
        raw = raw_fitness_function(ind, predictions)
        max_fitness = config_manager.get_param("fitness").get("max_fitness")

        ind.fitness = max_fitness - raw

        return ind.fitness


def adjusted_fitness_function(ind, predictions=None):
    assert isinstance(ind, SyntaxTree), "Individual is not an instance of SyntaxTree"

    standard = standardised_fitness_function(ind, predictions)

    ind.fitness = 1 / (1 + standard)

    return ind.fitness


def normalised_fitness_function(ind, predictions=None, pop=None):
    assert isinstance(ind, SyntaxTree), "Individual is not an instance of SyntaxTree"

    adjusted_fitness = adjusted_fitness_function(ind, predictions)

    copy_pop = None

    if pop is None:
        population = config_manager.get_config("population")
        copy_pop = deepcopy(population)
    else:
        copy_pop = deepcopy(pop)

    for individual in copy_pop.individuals:
        individual.fitness = adjusted_fitness_function(individual, individual.predict())

    sum_fitness = sum([individual.fitness for individual in copy_pop.individuals])

    ind.fitness = adjusted_fitness / sum_fitness

    return ind.fitness


def hit_rate_fitness_function(ind, predictions=None):
    assert isinstance(ind, SyntaxTree), "Individual is not an instance of SyntaxTree"

    if predictions is None:
        predictions = ind.predict()

    targets = config_manager.get_config("data").get("train_data").target

    num_hits = 0

    for i in range(len(predictions)):
        # print("PREDICTION: ", predictions[i])
        # print("TARGET: ", targets.iloc[i])
        # print("ERROR: ", error_function(predictions[i], targets.iloc[i]))
        error = error_function(predictions[i], targets.iloc[i])
        if error == 0:
            num_hits += 1

    ind.fitness = num_hits / len(predictions)

    return ind.fitness
