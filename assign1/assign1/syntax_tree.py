import math
import pandas as pd
import numpy as np
from assign1.utils import SingletonRandom
from assign1.config_classes.config_manager import ConfigurationManager

operators = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]
config_manager = ConfigurationManager()


class Node:
    _id_counter = 0

    def __init__(self, value):
        self.id = Node._id_counter
        Node._id_counter += 1
        self.value = value
        self.children = []
        self.fitness = None

    def is_leaf(self):
        return len(self.children) == 0

    def walk(self):
        # if self.is_leaf():
        #     yield self
        # else:
        #     for child in self.children:
        #         assert isinstance(child, Node), "Child is not an instance of Node"
        #         yield child.walk()
        yield self
        for child in self.children:
            assert isinstance(child, Node), "Child is not an instance of Node"
            yield from child.walk()

    def __str__(self):
        if self.is_leaf():
            return str(self.id) + ":" + str(self.value)
        return f"({self.id}:{self.value} {', '.join(str(child) for child in self.children)})"


class SyntaxTree:
    def __init__(self, root):
        assert isinstance(root, Node), "Root is not an instance of Node"
        self.root = root
        self.fitness = None

    def __str__(self):
        return str(self.root)

    def walk(self):
        yield from self.root.walk()

    def get_depth(self):
        def get_depth_helper(node):
            if node.is_leaf():
                return 1
            return 1 + max(get_depth_helper(child) for child in node.children)

        return get_depth_helper(self.root)

    def to_array(self):
        def to_array_helper(node):
            if node.is_leaf():
                return [node]

            return [node] + [
                item for child in node.children for item in to_array_helper(child)
            ]

        return to_array_helper(self.root)

    def evaluate(self, node, vals: np.array):
        # print("VALS: ", vals)
        assert isinstance(node, Node), "Node is not an instance of Node"
        if node.is_leaf():
            return float(vals[node.value])

        child_evals = []
        for i, child in enumerate(node.children):
            child_evals.append(self.evaluate(child, vals))

        if node.value == "+":
            sum_val = 0
            for j in child_evals:
                sum_val += j
            return sum_val

        if node.value == "-":
            sub_val = child_evals[0]
            for j in child_evals[1:]:
                sub_val -= j
            return sub_val

        if node.value == "*":
            prod_val = 1
            for j in child_evals:
                prod_val *= j
            return prod_val

        if node.value == "/":
            div_val = child_evals[0]
            for j in child_evals[1:]:
                if j == 0:
                    return 0
                div_val /= j
            return div_val

        if node.value == "sin":
            return math.sin(child_evals[0])

        if node.value == "cos":
            return math.cos(child_evals[0])

        if node.value == "exp":
            return math.exp(child_evals[0])

        if node.value == "log":
            return math.log(child_evals[0]) if child_evals[0] > 0 else 0

    def predict(self, vals: dict = None):
        if vals is None:
            vals: pd.Dataframe = config_manager.get_config("data").get("train_data")

        predictions = []
        for i in range(len(vals)):
            # print("VALS: ", vals.iloc[i])
            predictions.append(self.evaluate(self.root, vals.iloc[i]))

        return predictions

    @staticmethod
    def generate_random_leaf():
        randomboi = SingletonRandom()

        input_vals = config_manager.get_config("data").get("train_data").columns
        input_vals = input_vals.drop("target")

        return Node(randomboi.choice(input_vals))

    @staticmethod
    def generate_random_tree(depth: int, max_depth: int):

        if depth == max_depth:
            return SyntaxTree.generate_random_leaf()

        randomboi = SingletonRandom()

        input_vals = config_manager.get_config("data").get("train_data").columns
        input_vals = input_vals.drop("target").to_list()

        node_values = input_vals + operators

        # print("NV: ", node_values)

        operator = randomboi.choice(node_values)

        # print("OP: ", operator)

        root = None

        if operator in ["+", "-", "*", "/"]:
            root = Node(operator)
            root.children.append(SyntaxTree.generate_random_tree(depth + 1, max_depth))
            root.children.append(SyntaxTree.generate_random_tree(depth + 1, max_depth))
        elif operator in ["sin", "cos", "exp", "log"]:
            root = Node(operator)
            root.children.append(SyntaxTree.generate_random_tree(depth + 1, max_depth))
        else:
            root = Node(operator)

        return root

        # return Node("x")

    @staticmethod
    def generate_random_tree_grow(max_depth: int = 0):
        randomboi = SingletonRandom()

        if max_depth == 0:
            max_depth = config_manager.get_param("max_depth")

        operator = randomboi.choice(operators)

        root = Node(operator)

        if operator in ["+", "-", "*", "/"]:
            root.children.append(SyntaxTree.generate_random_tree(2, max_depth))
            root.children.append(SyntaxTree.generate_random_tree(2, max_depth))
        elif operator in ["sin", "cos", "exp", "log"]:
            root.children.append(SyntaxTree.generate_random_tree(2, max_depth))

        return SyntaxTree(root)

    @staticmethod
    def generate_random_tree_full(max_depth: int = 0):
        if max_depth == 0:
            max_depth = config_manager.get_param("max_depth")

        def grt_full_helper(depth: int, max_depth: int):
            if depth == max_depth:
                return SyntaxTree.generate_random_leaf()

            operator = randomboi.choice(operators)

            root = Node(operator)

            if operator in ["+", "-", "*", "/"]:
                root.children.append(grt_full_helper(depth + 1, max_depth))
                root.children.append(grt_full_helper(depth + 1, max_depth))
            elif operator in ["sin", "cos", "exp", "log"]:
                root.children.append(grt_full_helper(depth + 1, max_depth))

            return root

        randomboi = SingletonRandom()

        operator = randomboi.choice(operators)

        root = Node(operator)

        if operator in ["+", "-", "*", "/"]:
            root.children.append(grt_full_helper(2, max_depth))
            root.children.append(grt_full_helper(2, max_depth))
        elif operator in ["sin", "cos", "exp", "log"]:
            root.children.append(grt_full_helper(2, max_depth))

        return SyntaxTree(root)
