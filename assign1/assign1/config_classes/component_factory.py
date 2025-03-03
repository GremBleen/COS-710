from assign1.config_classes.config_manager import ConfigurationManager
from assign1.config_classes.plugin_manager import PluginManager
from assign1.population import Population
from assign1.syntax_tree import SyntaxTree

config_manager = ConfigurationManager()
plugin_manager = PluginManager()


class ComponentFactory:
    def initialisation_method(self):
        population_param = config_manager.get_param("population")
        method = population_param.get("method")

        population = Population()

        if method == "grow":
            plugin_manager.execute_plugin("initialisation_population_grow", population)
        elif method == "full":
            plugin_manager.execute_plugin("initialisation_population_full", population)
        elif method == "ramped":
            plugin_manager.execute_plugin(
                "initialisation_population_ramped", population
            )
        else:
            raise ValueError(f"Invalid initialisation method: {method}")

        return population

    def selection_method(self, population=None):
        selection_param = config_manager.get_param("selection")
        method = selection_param.get("method")

        if method == "fitness_proportionate":
            return plugin_manager.execute_plugin(
                "selection_fitness_proportionate", population
            )
        elif method == "tournament":
            return plugin_manager.execute_plugin("selection_tournament", population)
        else:
            raise ValueError(f"Invalid selection method: {method}")
        
    def genetic_operator_method(self):
        genetic_param = config_manager.get_param("genetic_operators")
        method = genetic_param.get("method")

        if method == "crossover_mutation":
            return plugin_manager.execute_plugin("genetic_operator_crossover_mutation")
        else:
            raise ValueError(f"Invalid genetic operator method: {method}")

    def crossover_method(self):
        crossover_param = config_manager.get_param("genetic_operators").get("crossover")
        method = crossover_param.get("method")

        if method == "single_point":
            return plugin_manager.execute_plugin("crossover_single_point")
        elif method == "two_point":
            raise NotImplementedError
            # return plugin_manager.execute_plugin("crossover_two_point")
        elif method == "uniform":
            raise NotImplementedError
            # return plugin_manager.execute_plugin("crossover_uniform")
        else:
            raise ValueError(f"Invalid crossover method: {method}")

    def mutation_method(self):
        mutation_param = config_manager.get_param("genetic_operators").get("mutation")
        method = mutation_param.get("method")

        if method == "point":
            raise NotImplementedError
            # return plugin_manager.execute_plugin("mutation_point")
        elif method == "subtree":
            return plugin_manager.execute_plugin("mutation_subtree")
        else:
            raise ValueError(f"Invalid mutation method: {method}")

    # def replacement_method(self):
    #     raise NotImplementedError

    # def termination_method(self):
    #     raise NotImplementedError

    def fitness_method(self, ind, predictions=None):
        assert isinstance(
            ind, SyntaxTree
        ), "Individual is not an instance of SyntaxTree"

        fitness_param = config_manager.get_param("fitness")
        method = fitness_param.get("method")

        if method == "raw":
            return plugin_manager.execute_plugin("fitness_raw", ind, predictions)
        elif method == "standardised":
            return plugin_manager.execute_plugin(
                "fitness_standardised", ind, predictions
            )
        elif method == "adjusted":
            return plugin_manager.execute_plugin("fitness_adjusted", ind, predictions)
        elif method == "normalised":
            return plugin_manager.execute_plugin("fitness_normalised", ind, predictions)
        elif method == "hit_rate":
            return plugin_manager.execute_plugin("fitness_hit_rate", ind, predictions)
        else:
            raise ValueError(f"Invalid fitness method: {method}")
