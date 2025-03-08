import math
import pandas as pd
import numpy as np
from termcolor import colored
from assign1.utils import SingletonRandom
from assign1.config_classes.config_manager import ConfigurationManager

operators = [
    "const",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "sqrt",
    "log",
    "exp",
    "max",
    "ifleq",
    "data",
    "avg",
]

operations = {
    "const": lambda p: p,
    "data": lambda p, inp: inp[p],
    "add": lambda p, inp: p[0].evaluate(inp) + p[1].evaluate(inp),
    "sub": lambda p, inp: p[0].evaluate(inp) - p[1].evaluate(inp),
    "mul": lambda p, inp: p[0].evaluate(inp) * p[1].evaluate(inp),
    "div": lambda p, inp: (lambda x, y: x / y if y != 0 else 0)(
        p[0].evaluate(inp), p[1].evaluate(inp)
    ),
    "pow": lambda p, inp: (
        lambda x, y: (
            (
                lambda n, m: (
                    0
                    if (n < 0 and not float(m).is_integer()) or (n == 0 and m < 0)
                    else float(n) ** float(m)
                )
            )(x, y)
            if y != 0
            else 1
        )
    )(p[0].evaluate(inp), p[1].evaluate(inp)),
    "sqrt": lambda p, inp: (lambda x: math.sqrt(x) if x >= 0 else 0)(
        p[0].evaluate(inp)
    ),
    "log": lambda p, inp: (lambda x: math.log(x, 2) if x > 0 else 0)(
        p[0].evaluate(inp)
    ),
    "exp": lambda p, inp: (lambda x: x if x != float("inf") else 0)(
        math.exp(p[0].evaluate(inp))
    ),
    "max": lambda p, inp: max(p[0].evaluate(inp), p[1].evaluate(inp)),
    "ifleq": lambda p, inp: (
        p[2].evaluate(inp)
        if p[0].evaluate(inp) <= p[1].evaluate(inp)
        else p[3].evaluate(inp)
    ),
    "avg": lambda p, inp: (p[0].evaluate(inp) + p[1].evaluate(inp)) / 2,
}

op_branches = {
    "const": 0,
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "pow": 2,
    "sqrt": 1,
    "log": 1,
    "exp": 1,
    "max": 2,
    "ifleq": 4,
    "data": 0,
    "avg": 2,
}

# calls to op_constructors should be in the form of op_constructors["add"]("generation_function", depth, max_depth)
op_constructors = {
    4: lambda op, gen_func, depth, max_depth: Node(op, children=[gen_func(depth, max_depth), gen_func(depth, max_depth), gen_func(depth, max_depth), gen_func(depth, max_depth)]),
    3: lambda op, gen_func, depth, max_depth: Node(op, children=[gen_func(depth, max_depth), gen_func(depth, max_depth), gen_func(depth, max_depth)]),
    2: lambda op, gen_func, depth, max_depth: Node(op, children=[gen_func(depth, max_depth), gen_func(depth, max_depth)]),
    1: lambda op, gen_func, depth, max_depth: Node(op, children=[gen_func(depth, max_depth)]),
    0: lambda op, gen_func, depth, max_depth: SyntaxTree.generate_random_const() if op == "const" else SyntaxTree.generate_random_data(),
}

config_manager = ConfigurationManager()


class Node:
    _id_counter = 0

    def __init__(self, op, value=None, children=None):
        self.id = Node._id_counter
        Node._id_counter += 1
        self.op = op
        self.value = value
        self.children = children if children is not None else []
        self.fitness = None

    def is_leaf(self):
        return self.op in ["const", "data"]

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

    def evaluate(self, vals: np.array):
        try:
            ans = 0
            if self.is_leaf():
                if self.op == "const":
                    ans = operations[self.op](self.value)
                elif self.op == "data":
                    ans = operations[self.op](self.value, vals)
            else:
                ans = operations[self.op](self.children, vals)

            if ans == float("inf") or math.isnan(ans):
                return 0
            else:
                return ans
        except OverflowError:
            return 0

    def __str__(self):
        id_str = str(self.id)
        if self.is_leaf():
            return id_str + ":" + str(self.value)
        return (
            f"({id_str}:{self.op} {', '.join(str(child) for child in self.children)})"
        )

    def to_string_colored(self):
        id_str = colored(str(self.id), "red")
        if self.is_leaf():
            return id_str + ":" + str(self.value)
        return (
            f"({id_str}:{self.op} {', '.join(child.to_string_colored() for child in self.children)})"
        )

    def to_string(self):
        id_str = str(self.id)
        if self.is_leaf():
            return id_str + ":" + str(self.value)
        return (
            f"({id_str}:{self.op} {', '.join(str(child) for child in self.children)})"
        )


class SyntaxTree:
    def __init__(self, root):
        assert isinstance(root, Node), "Root is not an instance of Node"
        self.root = root
        self.fitness = None

    def __str__(self):
        return str(self.root)
    
    def to_string_colored(self):
        return self.root.to_string_colored()
    
    def to_string(self):
        return self.root.to_string()
    
    def to_json(self):
        return {"tree": str(self), "fitness": self.fitness}

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

    def evaluate(self, vals: np.array):
        return self.root.evaluate(vals)

    def predict(self, vals: dict = None):
        if vals is None:
            vals: pd.Dataframe = config_manager.get_config("data").get("train_data")

        predictions = []
        for i in range(len(vals)):
            # print("VALS: ", vals.iloc[i])
            predictions.append(self.evaluate(vals.iloc[i]))

        return predictions

    @staticmethod
    def generate_random_data():
        randomboi = SingletonRandom()

        input_vals = config_manager.get_config("data").get("train_data").columns
        input_vals = input_vals.drop("target")

        return Node("data", randomboi.choice(input_vals))

    @staticmethod
    def generate_random_const():
        randomboi = SingletonRandom()

        return Node("const", randomboi.randint(-10, 10))

    @staticmethod
    def generate_random_leaf():
        randomboi = SingletonRandom()

        if randomboi.random() < 0.5:
            return SyntaxTree.generate_random_const()
        else:
            return SyntaxTree.generate_random_data()

    @staticmethod
    def generate_random_tree(depth: int, max_depth: int):
        if depth == max_depth:
            return SyntaxTree.generate_random_leaf()

        randomboi = SingletonRandom()

        # print("NV: ", node_values)

        valid_ops = operators.copy()
        valid_ops.remove("const")

        operator = randomboi.choice(valid_ops)

        # print("OP: ", operator)

        root = op_constructors[op_branches[operator]](operator, SyntaxTree.generate_random_tree, depth + 1, max_depth)

        return root

        # return Node("x")

    @staticmethod
    def generate_random_tree_grow(max_depth: int = 0):
        randomboi = SingletonRandom()

        if max_depth == 0:
            max_depth = config_manager.get_param("max_depth")

        root_ops = operators.copy()
        root_ops.remove("const")
        root_ops.remove("data")

        operator = randomboi.choice(root_ops)

        # NOTE CALLING THIS WITH 2 INSTEAD OF 1 will cause the maximum recursion depth to be exceeded
        root = op_constructors[op_branches[operator]](operator, SyntaxTree.generate_random_tree, 1, max_depth)

        return SyntaxTree(root)

    @staticmethod
    def generate_random_tree_full(max_depth: int = 0):
        if max_depth == 0:
            max_depth = config_manager.get_param("max_depth")

        def grt_full_helper(depth: int, max_depth: int):
            if depth == max_depth:
                return SyntaxTree.generate_random_leaf()

            root_ops = operators.copy()
            root_ops.remove("const")
            root_ops.remove("data")

            operator = randomboi.choice(root_ops)

            root = op_constructors[op_branches[operator]](operator, grt_full_helper, depth + 1, max_depth)

            return root

        randomboi = SingletonRandom()

        root_ops = operators.copy()
        root_ops.remove("const")
        root_ops.remove("data")

        operator = randomboi.choice(root_ops)

        # NOTE CALLING THIS WITH 2 INSTEAD OF 1 will cause the maximum recursion depth to be exceeded
        root = op_constructors[op_branches[operator]](operator, grt_full_helper, 1, max_depth)

        return SyntaxTree(root)
