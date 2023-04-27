import argparse
from importlib import import_module


def get_func(func_name):
    """An easy call function to get Module by name.
    """
    if func_name is None:
        return None
    parts = func_name.split('.')
    if len(parts) == 1:
        return globals()[parts[0]]
    module_name = '.'.join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])


def parse_args():
    """An easy method get config file.
    """
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument(
        '--cfg', help='experiment configure file path', required=True, type=str)  # noqa

    return parser.parse_args()
