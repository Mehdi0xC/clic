import yaml
import json
from pathlib import Path

import argparse

def parse_overrides():
    parser = argparse.ArgumentParser(description='Override config values.')
    parser.add_argument('--overrides', type=str, help='Overrides in the form key=value,key2=value2,...')

    args = parser.parse_args()
    overrides = {}

    if args.overrides:
        pairs = args.overrides.split(',')
        for pair in pairs:
            key, value = pair.split('=')
            overrides[key] = value

    return overrides


class Config(dict):
    """
    A dictionary subclass with attribute-style access and automatic 
    recursive transformation of nested dictionaries.
    """
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__()
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    if isinstance(v, dict):
                        v = Config(v)
                    if isinstance(v, list):
                        self.__convert(v)
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    v = Config(v)
                elif isinstance(v, list):
                    self.__convert(v)
                self[k] = v

    def __convert(self, v):
        for elem in range(0, len(v)):
            if isinstance(v[elem], dict):
                v[elem] = Config(v[elem])
            elif isinstance(v[elem], list):
                self.__convert(v[elem])

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Config, self).__delitem__(key)
        del self.__dict__[key]

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, indent=4)




def _parse_value(value: str):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value




def load_config(argv) -> Config:
    """
    Load a YAML configuration file into a Config object. Print the 
    'message' attribute if it exists.
    """
    config_name = argv[1]
    config_path = Path("configs") / f"{config_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"No configuration file found at {config_path}")
    
    print(f"Loading config from {config_path}")
    with open(config_path) as config_file:
        config_dict = yaml.safe_load(config_file)
    
    config = Config(config_dict)
    
    if 'message' in config:
        print(config.message)
    
    for arg in argv[2:]:
        print(arg)
        if arg[0] == '-':
            continue
        key, value = arg.split('=')
        value = _parse_value(value)
        setattr(config, key, value)

    return config

