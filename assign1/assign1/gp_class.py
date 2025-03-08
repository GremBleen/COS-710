from assign1.config_classes.component_factory import ComponentFactory
from assign1.config_classes.config_manager import ConfigurationManager
from assign1.config_classes.plugin_manager import PluginManager
from assign1.utils import SingletonRandom
from assign1.population import Population
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
import concurrent.futures
import os
import time

config_manager = ConfigurationManager()
plugin_manager = PluginManager()
component_factory = ComponentFactory()

class GPClass:
    def __init__(self):
        self.train_time = 0
        self.test_time = 0
        self.best_individual = None

    def fit(self, data, config_file="default_config.json"):
        config_manager.load_configs_from_file(config_file)
        seed = config_manager.get_param("seed")

        randomboi = SingletonRandom()
        randomboi.seed(seed)

        train, test = train_test_split(data, test_size=0.2, random_state=seed)

        config_manager.set_config("data", {"train_data": train, "test_data": test})
        config_manager.set_config("generations", [])

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

    def train(self):
        start_time = time.time()

        self.set_up_population()

        # print(config_manager.get_config("population").individuals[0].to_string_colored())

        num_generations = config_manager.get_param("stopping_criteria").get("max_generations")
        for i in range(num_generations):
            new_population = component_factory.genetic_operator_method()

            self.evaluate_fitness(new_population, "train")
            # print(new_population.individuals[0].to_string_colored())
            config_manager.set_config("population", new_population)

            generation = {
                "num": i + 1,
                "population": new_population.to_json(),
                "average_fitness": new_population.average_fitness(),
                "best_individual": new_population.best_individual().to_json(),
            }

            print(f"Generation {i + 1} - Average Fitness: {generation['average_fitness']}")

            generations = config_manager.get_config("generations")
            generations.append(generation)
            config_manager.set_config("generations", generations)

        end_time = time.time()
        self.train_time = end_time - start_time
        print(f"Training completed in {end_time - start_time} seconds")

    def test(self):
        start_time = time.time()

        self.set_up_population()

        population = config_manager.get_config("population")

        self.evaluate_fitness(population, "test")
        config_manager.set_config("population", population)

        run_results = {
            "population": population.to_json(),
            "average_fitness": population.average_fitness(),
            "best_individual": population.best_individual().to_json(),
            "train_time": self.train_time,
            "test_time": self.test_time,
        }

        config_manager.set_config("run_results", run_results)

        print(
            f"Testing - Average Fitness: {run_results['average_fitness']}"
        )

        end_time = time.time()
        self.test_time = end_time - start_time
        print(f"Testing completed in {end_time - start_time} seconds")

        run_results["test_time"] = self.test_time

        config_manager.save_configs_to_file("results.json")

    def set_up_population(self):
        population = component_factory.initialisation_method()

        config_manager.set_config("population", population)

        self.evaluate_fitness(population)

        generation0 = {
            "num": 0,
            "population": population.to_json(),
            "average_fitness": population.average_fitness(),
            "best_individual": population.best_individual().to_json(),
        }

        generations = config_manager.get_config("generations")
        generations.append(generation0)
        config_manager.set_config("generations", generations)

        # print("GEN: ",config_manager.get_config("generations"))

    @staticmethod
    def evaluate_individual(ind, data_type):
        if data_type == "train":
            data = config_manager.get_config("data").get("train_data")
        elif data_type == "test":
            data = config_manager.get_config("data").get("test_data")

        comp_fact = ComponentFactory()
        return comp_fact.fitness_method(ind, ind.predict(data))

    def evaluate_fitness(self, population, data_type = "train"):
        # Use ProcessPoolExecutor to parallelise fitness evaluation across all cpu cores
        num_workers = os.cpu_count()
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.evaluate_individual, ind, data_type): ind for ind in population.individuals}

            for future in concurrent.futures.as_completed(futures):
                ind = futures[future]
                ind.fitness = future.result()

    # def evaluate_fitness(self, population):
    #     for ind in population.individuals:
    #         component_factory.fitness_method(ind, ind.predict())
