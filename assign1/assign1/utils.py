import random


class SingletonRandom(object):
    def __new__(cls, seed=None):
        if not hasattr(cls, "_instance"):
            cls._instance = super(SingletonRandom, cls).__new__(cls)
            cls._instance._random = random.Random(seed)
            cls._instance.seed(seed)
            print(f"Random seed: {seed}")

        return cls._instance

    def randint(self, a, b):
        return self._random.randint(a, b)

    def random(self):
        return self._random.random()

    def choice(self, seq):
        return self._random.choice(seq)

    def sample(self, population, k):
        return self._random.sample(population, k)

    def seed(self, seed):
        self._random.seed(seed)

    def get_seed(self):
        return self._random.seed
