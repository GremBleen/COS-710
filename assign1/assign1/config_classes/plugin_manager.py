class SingletonMeta(type):
    """
    A Singleton metaclass that creates a single instance of a class.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class PluginManager(metaclass=SingletonMeta):
    def __init__(self):
        self.plugins = {}

    def register_plugin(self, name, func):
        self.plugins[name] = func

    def get_plugin(self, name):
        return self.plugins.get(name)

    def execute_plugin(self, name, *args, **kwargs):
        if name in self.plugins:
            return self.plugins[name](*args, **kwargs)
        else:
            raise ValueError(f"Plugin {name} not found")
