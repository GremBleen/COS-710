from assign1.syntax_tree import Node, SyntaxTree
from assign1.utils import SingletonRandom
from assign1.config_manager import ConfigurationManager
from assign1.plugin_manager import PluginManager
from assign1.component_factory import ComponentFactory
from sklearn.model_selection import train_test_split
from assign1.population import Population
import pandas as pd

data = pd.read_csv("192_vineyard.tsv", sep="\t")

randomboi = SingletonRandom(40)

print(data)

print(data.columns.drop("target"))

config_manager = ConfigurationManager()

train, test = train_test_split(data, test_size=0.2, random_state=40)

config_manager.set_config("data", {"train_data": train, "test_data": test})
config_manager.set_param("max_depth", 5)

config_manager.set_param("population", {
    "method": "grow",
    "size": 10
})

plugin_manager = PluginManager()
component_factory = ComponentFactory()

plugin_manager.register_plugin("initialisation_population_grow", Population.ini_population_grow)
plugin_manager.register_plugin("initialisation_population_full", Population.ini_population_full)

population = component_factory.initialisation_method()

tree = population.individuals[0]

for ind in population.individuals:
    print("IND: ", ind)
    print("DEPTH: ", ind.get_depth())
