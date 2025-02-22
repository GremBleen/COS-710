import json
from assign1.state import State

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


class ConfigurationManager(metaclass=SingletonMeta):
    def __init__(self):
        self.config = {
            "parameters": {},
            "data": {},
            "generations": {}
        }

    def set_config(self, key, value):
        self.config[key] = value

    def set_param(self, key, value):
        params = self.config.get("parameters")
        params[key] = value
        self.config["parameters"] = params

    def get_config(self, key):
        return self.config.get(key)
    
    def get_param(self, key):
        return self.config.get("parameters").get(key)
    
    def get_configs(self):
        return self.config
    
    def get_params(self):
        return self.config.get("parameters")

    def update_configs(self, new_configs):
        self.config.update(new_configs)

    def update_params(self, new_params):
        params = self.config.get("parameters")
        params.update(new_params)
        self.config["parameters"] = params

    def update_configs_deep(self, **kwargs):
        for key, value in kwargs.items():
            if (
                key in self.config
                and isinstance(self.config[key], dict)
                and isinstance(value, dict)
            ):
                self.config[key].update(value)
            else:
                self.config[key] = value

    def save_configs(self):
        return State(self.config)

    def load_configs(self, state):
        self.config = state.configg

    def load_configs_from_file(self, file_path):
        with open(file_path, "r") as file:
            self.config = json.load(file)

    def __str__(self):
        return json.dumps(self.config, indent=4)
