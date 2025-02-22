from assign1.syntax_tree import Node, SyntaxTree
from assign1.utils import SingletonRandom
from assign1.config_classes.config_manager import ConfigurationManager
from assign1.config_classes.plugin_manager import PluginManager
from assign1.config_classes.component_factory import ComponentFactory
from sklearn.model_selection import train_test_split
from assign1.population import Population
from assign1.fitness_functions import error_function, raw_fitness_function, standardised_fitness_function, adjusted_fitness_function, normalised_fitness_function, hit_rate_fitness_function
import pandas as pd
import numpy as np

randomboi = SingletonRandom(40)
config_manager = ConfigurationManager()
plugin_manager = PluginManager()
component_factory = ComponentFactory()

data = pd.read_csv("192_vineyard.tsv", sep="\t")

train, test = train_test_split(data, test_size=0.2, random_state=40)

config_manager.load_configs_from_file("param_template.json")

config_manager.set_config("data", {"train_data": train, "test_data": test})

plugin_manager.register_plugin("initialisation_population_grow", Population.ini_population_grow)
plugin_manager.register_plugin("initialisation_population_full", Population.ini_population_full)
plugin_manager.register_plugin("initialisation_population_ramped", Population.ini_population_ramped)

plugin_manager.register_plugin("fitness_raw", raw_fitness_function)
plugin_manager.register_plugin("fitness_standardised", standardised_fitness_function)
plugin_manager.register_plugin("fitness_adjusted", adjusted_fitness_function)
plugin_manager.register_plugin("fitness_normalised", normalised_fitness_function)
plugin_manager.register_plugin("fitness_hit_rate", hit_rate_fitness_function)


population = component_factory.initialisation_method()

tree = population.individuals[0]

predictions = tree.predict()

print(component_factory.fitness_method(tree, predictions))