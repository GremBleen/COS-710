from assign1.config_manager import ConfigurationManager
from assign1.plugin_manager import PluginManager
from assign1.population import Population

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
        elif method == "ramped_half_and_half":
            plugin_manager.execute_plugin("initialisation_population_ramped_half_and_half", population)
        else:
            raise ValueError(f"Invalid initialisation method: {method}")
        
        return population

    def selection_method(self):
        raise NotImplementedError

    def crossover_method(self):
        raise NotImplementedError

    def mutation_method(self):
        raise NotImplementedError

    # def replacement_method(self):
    #     raise NotImplementedError

    # def termination_method(self):
    #     raise NotImplementedError

    def fitness_method(self):
        raise NotImplementedError
