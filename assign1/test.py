from assign1.syntax_tree import Node, SyntaxTree, op_constructors, op_branches
from assign1.utils import SingletonRandom
from assign1.config_classes.config_manager import ConfigurationManager
from assign1.config_classes.plugin_manager import PluginManager
from assign1.config_classes.component_factory import ComponentFactory
from sklearn.model_selection import train_test_split
from assign1.population import Population
from assign1.fitness_functions import error_function, raw_fitness_function, standardised_fitness_function, adjusted_fitness_function, normalised_fitness_function, hit_rate_fitness_function
from assign1.selection_method import fitness_proportionate_selection, tournament_selection
from assign1.genetic_operators import single_point_crossover, subtree_mutation, crossover_mutation_genetic_operator, prune_tree
import pandas as pd
import numpy as np

seed = 20

randomboi = SingletonRandom(seed)
config_manager = ConfigurationManager()
plugin_manager = PluginManager()
component_factory = ComponentFactory()

data = pd.read_csv("192_vineyard.tsv", sep="\t")

train, test = train_test_split(data, test_size=0.2, random_state=seed)

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

plugin_manager.register_plugin("selection_fitness_proportionate", fitness_proportionate_selection)
plugin_manager.register_plugin("selection_tournament", tournament_selection)

plugin_manager.register_plugin("genetic_operator_crossover_mutation", crossover_mutation_genetic_operator)
plugin_manager.register_plugin("crossover_single_point", single_point_crossover)
plugin_manager.register_plugin("mutation_subtree", subtree_mutation)

population = component_factory.initialisation_method()

config_manager.set_config("population", population)

test_node = op_constructors[op_branches["add"]]("add", SyntaxTree.generate_random_tree, 1, 3)
print("TestNode: ", test_node)

test_tree = SyntaxTree.generate_random_tree_grow(3)
print("TestTree: ", test_tree)

for ind in population.individuals:
    component_factory.fitness_method(ind, ind.predict())

tree1 = component_factory.selection_method(population)
tree2 = component_factory.selection_method(population)
tree3 = component_factory.selection_method(population)
rand_tree = SyntaxTree(SyntaxTree.generate_random_tree(1, 4))

print("TREE1 ", tree1)
print("TREE3 ", tree3)

child1, child2 = single_point_crossover(tree1, tree3)

print("CHILD1: ",child1)
print("CHILD2: ",child2)

# print("PRUNE", prune_tree(tree))
# print(tree.fitness)