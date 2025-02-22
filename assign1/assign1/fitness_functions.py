import numpy as np
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

    return np.sum(error)


def standardised_fitness_function(ind, predictions=None):
    assert isinstance(ind, SyntaxTree), "Individual is not an instance of SyntaxTree"

    better_fitness = config_manager.get_param("fitness").get("better_fitness")
    if better_fitness == "low":
        return raw_fitness_function(ind, predictions)
    elif better_fitness == "high":
        raw = raw_fitness_function(ind, predictions)
        max_fitness = config_manager.get_param("fitness").get("max_fitness")

        return max_fitness - raw


def adjusted_fitness_function(ind, predictions=None):
    assert isinstance(ind, SyntaxTree), "Individual is not an instance of SyntaxTree"

    standard = standardised_fitness_function(ind, predictions)

    return (1 / (1 + standard))


def normalised_fitness_function(ind, predictions=None):
    raise NotImplementedError

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

    return num_hits / len(predictions)