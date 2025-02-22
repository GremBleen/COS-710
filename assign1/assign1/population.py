from assign1.config_manager import ConfigurationManager 
from assign1.syntax_tree import SyntaxTree

config_manager = ConfigurationManager()

class Population:
    def __init__(self):
        self.individuals = []

    @staticmethod
    def ini_population_grow(pop):
        assert isinstance(pop, Population), "Population is not an instance of Population"
        for i in range(config_manager.get_param("population").get("size")):
            pop.individuals.append(SyntaxTree.generate_random_tree_grow())
        return pop
    
    @staticmethod
    def ini_population_full(pop):
        assert isinstance(pop, Population), "Population is not an instance of Population"
        for i in range(config_manager.get_param("population").get("size")):
            pop.individuals.append(SyntaxTree.generate_random_tree_full())
        return pop