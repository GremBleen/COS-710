import math
from assign1.utils import SingletonRandom
from assign1.config_manager import ConfigurationManager

operators = ["+", "-", "*", "/", "sin", "cos", "exp", "log"]
config_manager = ConfigurationManager()

class Node:

    def __init__(self, value):
        self.value = value
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

    def walk(self):
        if self.is_leaf():
            yield self
        else:
            for child in self.children:
                print(child)
                assert isinstance(child, Node), "Child is not an instance of Node"
                yield child.walk()

    def __str__(self):
        if self.is_leaf():
            return str(self.value)
        return f"{self.value} ({', '.join(str(child) for child in self.children)})"


class SyntaxTree:
    def __init__(self, root):
        assert isinstance(root, Node), "Root is not an instance of Node"
        self.root = root
        self.fitness = None

    def __str__(self):
        return str(self.root)

    def get_depth(self):
        def get_depth_helper(node):
            if node.is_leaf():
                return 1
            return 1 + max(get_depth_helper(child) for child in node.children)

        return get_depth_helper(self.root)

    def evaluate(self, node, vals: dict):
        assert isinstance(node, Node), "Node is not an instance of Node"
        if node.is_leaf():
            return vals[node.value]

        child_evals = []
        for i, child in enumerate(node.children):
            child_evals = self.evaluate(child, vals)

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
            return math.log(child_evals[0])

    def predict(self, vals: dict):
        # TODO: Implement this method
        raise NotImplementedError("TODO")

    @staticmethod
    def generate_random_leaf():
        randomboi = SingletonRandom()

        input_vals = config_manager.get_config("data").get("train_data").columns
        input_vals = input_vals.drop("target")

        return Node(randomboi.choice(input_vals))

    @staticmethod
    def generate_random_tree(depth: int):

        if depth == config_manager.get_param("max_depth"):
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
            root.children.append(SyntaxTree.generate_random_tree(depth + 1))
            root.children.append(SyntaxTree.generate_random_tree(depth + 1))
        elif operator in ["sin", "cos", "exp", "log"]:
            root = Node(operator)
            root.children.append(SyntaxTree.generate_random_tree(depth + 1))
        else:
            root = Node(operator)

        return root

        # return Node("x")

    @staticmethod
    def generate_random_tree_grow():  # This function generates a tree of depth > 1 and < max_depth randomly
        randomboi = SingletonRandom()

        operator = randomboi.choice(operators)

        root = Node(operator)

        if operator in ["+", "-", "*", "/"]:
            root.children.append(SyntaxTree.generate_random_tree(2))
            root.children.append(SyntaxTree.generate_random_tree(2))
        elif operator in ["sin", "cos", "exp", "log"]:
            root.children.append(SyntaxTree.generate_random_tree(2))

        return SyntaxTree(root)

    @staticmethod
    def generate_random_tree_full():
        def grt_full_helper(depth: int):
            if depth == config_manager.get_param("max_depth"):
                return SyntaxTree.generate_random_leaf()

            operator = randomboi.choice(operators)

            root = Node(operator)

            if operator in ["+", "-", "*", "/"]:
                root.children.append(grt_full_helper(depth + 1))
                root.children.append(grt_full_helper(depth + 1))
            elif operator in ["sin", "cos", "exp", "log"]:
                root.children.append(grt_full_helper(depth + 1))

            return root

        randomboi = SingletonRandom()

        operator = randomboi.choice(operators)

        root = Node(operator)

        if operator in ["+", "-", "*", "/"]:
            root.children.append(grt_full_helper(2))
            root.children.append(grt_full_helper(2))
        elif operator in ["sin", "cos", "exp", "log"]:
            root.children.append(grt_full_helper(2))

        return SyntaxTree(root)
