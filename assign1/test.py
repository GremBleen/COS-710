import pandas as pd
from assign1.config_classes.config_manager import ConfigurationManager
from assign1.config_classes.component_factory import ComponentFactory
from assign1.gp_class import GPClass
from assign1.utils import SingletonRandom
from sklearn.model_selection import train_test_split
from assign1.fitness_functions import (
    raw_fitness_function,
    standardised_fitness_function,
    adjusted_fitness_function,
    normalised_fitness_function,
    hit_rate_fitness_function,
)
from assign1.selection_method import (
    fitness_proportionate_selection,
    tournament_selection,
)
from assign1.genetic_operators import (
    crossover_mutation_genetic_operator,
    single_point_crossover,
    subtree_mutation,
)
from assign1.config_classes.plugin_manager import PluginManager
import os
from assign1.population import Population
from assign1.syntax_tree import SyntaxTree

seed = 21
randomboi = SingletonRandom(seed=seed)

data = pd.read_csv("192_vineyard.tsv", sep="\t")

config_manager = ConfigurationManager()
plugin_manager = PluginManager()
component_factory = ComponentFactory()

config_file = "default_config.json"

train, test = train_test_split(data, test_size=0.2, random_state=seed)

config_manager.load_configs_from_file(config_file)

config_manager.set_config("data", {"train_data": train, "test_data": test})

plugin_manager.register_plugin(
    "initialisation_population_grow", Population.ini_population_grow
)
plugin_manager.register_plugin(
    "initialisation_population_full", Population.ini_population_full
)
plugin_manager.register_plugin(
    "initialisation_population_ramped", Population.ini_population_ramped
)

plugin_manager.register_plugin("fitness_raw", raw_fitness_function)
plugin_manager.register_plugin(
    "fitness_standardised", standardised_fitness_function
)
plugin_manager.register_plugin("fitness_adjusted", adjusted_fitness_function)
plugin_manager.register_plugin(
    "fitness_normalised", normalised_fitness_function
)
plugin_manager.register_plugin("fitness_hit_rate", hit_rate_fitness_function)

plugin_manager.register_plugin(
    "selection_fitness_proportionate", fitness_proportionate_selection
)
plugin_manager.register_plugin("selection_tournament", tournament_selection)

plugin_manager.register_plugin(
    "genetic_operator_crossover_mutation", crossover_mutation_genetic_operator
)
plugin_manager.register_plugin("crossover_single_point", single_point_crossover)
plugin_manager.register_plugin("mutation_subtree", subtree_mutation)

# tree = SyntaxTree.generate_random_tree_grow()
# print(tree.to_string_colored())

# pop = component_factory.initialisation_method()

# print(pop.individuals[0].to_string_colored())

# config_manager.set_config("population", pop)

# for ind in pop.individuals:
#     component_factory.fitness_method(ind, ind.predict())

# print(component_factory.genetic_operator_method().individuals[0].to_string_colored())


print(randomboi.randint(0, 10))
randomboi.seed(21)
print(randomboi.randint(0, 10))