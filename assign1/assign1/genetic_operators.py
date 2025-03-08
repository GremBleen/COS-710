from assign1.syntax_tree import SyntaxTree, Node
from assign1.utils import SingletonRandom
from assign1.config_classes.config_manager import ConfigurationManager
from assign1.config_classes.component_factory import ComponentFactory
from assign1.population import Population
from copy import deepcopy
import concurrent.futures
import os

config_manager = ConfigurationManager()
component_factory = ComponentFactory()


def single_point_crossover(ind1, ind2):  # Single point crossover
    assert isinstance(ind1, SyntaxTree), "ind1 must be an instance of SyntaxTree"
    assert isinstance(ind2, SyntaxTree), "ind2 must be an instance of SyntaxTree"

    randomboi = SingletonRandom()

    copy_ind1 = deepcopy(ind1)
    copy_ind2 = deepcopy(ind2)

    ind1_array = copy_ind1.to_array()
    ind2_array = copy_ind2.to_array()

    ind1_cop = randomboi.randint(
        1, len(ind1_array) - 1
    )  # Index minimum is 1 to avoid selecting the root node
    ind2_cop = randomboi.randint(1, len(ind2_array) - 1)

    child1 = swap_subtree(copy_ind1, ind1_array[ind1_cop].id, ind2_array[ind2_cop])
    child2 = swap_subtree(copy_ind2, ind2_array[ind2_cop].id, ind1_array[ind1_cop])

    return prune_tree(child1), prune_tree(child2)


def subtree_mutation(ind):
    assert isinstance(ind, SyntaxTree), "ind must be an instance of SyntaxTree"

    randomboi = SingletonRandom()

    copy_ind = deepcopy(ind)

    subtree = SyntaxTree.generate_random_tree(1, config_manager.get_param("max_depth"))

    ind_array = copy_ind.to_array()

    ind_cop = randomboi.randint(1, len(ind_array) - 1)

    mutated_ind = swap_subtree(copy_ind, ind_array[ind_cop].id, subtree)

    return prune_tree(mutated_ind)


def swap_subtree(ind, node_id, subtree):
    assert isinstance(ind, SyntaxTree), "ind must be an instance of SyntaxTree"
    assert isinstance(subtree, Node), "subtree must be an instance of Node"

    copy_ind = deepcopy(ind)
    subtree_copy = deepcopy(subtree)

    for node in copy_ind.walk():
        for i, child in enumerate(node.children):
            if child.id == node_id:
                node.children[i] = subtree_copy
                return copy_ind


def prune_tree(ind):
    def prune_tree_helper(depth, node):
        if depth > config_manager.get_param("max_depth"):
            return SyntaxTree.generate_random_leaf()
        else:
            for i, child in enumerate(node.children):
                node.children[i] = prune_tree_helper(depth + 1, child)
            return node

    # travserse the tree and replace the nodes that are deeper than the max depth with a leaf node
    assert isinstance(ind, SyntaxTree), "ind must be an instance of SyntaxTree"

    return SyntaxTree(prune_tree_helper(1, ind.root))


def crossover_stage(new_population):
    population = config_manager.get_config("population")

    # Crossover Stage
    # 1.1 Select two parents

    parent1 = component_factory.selection_method(population)
    parent2 = component_factory.selection_method(population)

    # 1.2 Perform crossover

    child1, child2 = component_factory.crossover_method(parent1, parent2)

    new_population.individuals.append(child1)
    new_population.individuals.append(child2)

    return new_population


def mutation_stage(new_population):
    population = config_manager.get_config("population")

    # Mutation Stage
    # 2.1 Select a parent

    parent = component_factory.selection_method(population)

    # 2.2 Perform mutation

    mutated_child = component_factory.mutation_method(parent)

    new_population.individuals.append(mutated_child)

    return new_population


def crossover_mutation_genetic_operator():
    new_population = Population()

    num_cross = int(
        config_manager.get_param("genetic_operators").get("crossover").get("rate")
        * config_manager.get_param("population").get("size")
    )
    num_mut = int(
        config_manager.get_param("genetic_operators").get("mutation").get("rate")
        * config_manager.get_param("population").get("size")
    )

    for _ in range(int(num_cross)):
        new_population = crossover_stage(new_population)

    for _ in range(int(num_mut)):
        new_population = mutation_stage(new_population)

    return new_population
