from math import ceil, floor
from assign1.config_classes.config_manager import ConfigurationManager
from assign1.utils import SingletonRandom
from assign1.syntax_tree import SyntaxTree

config_manager = ConfigurationManager()


class Population:
    def __init__(self):
        self.individuals = []

    @staticmethod
    def ini_population_grow(pop):
        assert isinstance(
            pop, Population
        ), "Population is not an instance of Population"
        for i in range(config_manager.get_param("population").get("size")):
            pop.individuals.append(SyntaxTree.generate_random_tree_grow())
        return pop

    @staticmethod
    def ini_population_full(pop):
        assert isinstance(
            pop, Population
        ), "Population is not an instance of Population"
        for i in range(config_manager.get_param("population").get("size")):
            pop.individuals.append(SyntaxTree.generate_random_tree_full())
        return pop

    @staticmethod
    def ini_population_ramped(pop):
        randomboi = SingletonRandom()

        assert isinstance(
            pop, Population
        ), "Population is not an instance of Population"
        max_depth = config_manager.get_param("max_depth")
        pop_size = config_manager.get_param("population").get("size")

        num_ind_per_depth = pop_size / (max_depth - 1)

        num_ind_per_depth = floor(num_ind_per_depth)

        # print("NUM IND PER DEPTH: ", num_ind_per_depth)

        for i in range(2, max_depth + 1):
            if num_ind_per_depth % 2 != 0:
                if randomboi.choice([True, False]):
                    # print("RFULL I: ", i)
                    pop.individuals.append(SyntaxTree.generate_random_tree_full(i))
                else:
                    # print("RGROW I: ", i)
                    pop.individuals.append(SyntaxTree.generate_random_tree_grow(i))

            for _ in range(num_ind_per_depth // 2):
                # print("FULL I: ", i)
                pop.individuals.append(SyntaxTree.generate_random_tree_full(i))

            for _ in range(num_ind_per_depth // 2):
                # print("GROW I: ", i)
                pop.individuals.append(SyntaxTree.generate_random_tree_grow(i))

        if len(pop.individuals) < pop_size:
            for _ in range(pop_size - len(pop.individuals)):
                if randomboi.choice([True, False]):
                    # print("OVER FULL")
                    pop.individuals.append(
                        SyntaxTree.generate_random_tree_full(max_depth)
                    )
                else:
                    # print("OVER GROW")
                    pop.individuals.append(
                        SyntaxTree.generate_random_tree_grow(max_depth)
                    )

        return pop
    
    def to_json(self):
        return [{"tree": str(ind), "fitness": ind.fitness} for ind in self.individuals]

    def average_fitness(self):
        return sum([ind.fitness for ind in self.individuals]) / len(self.individuals)

    def best_individual(self):
        return max(self.individuals, key=lambda x: x.fitness)
