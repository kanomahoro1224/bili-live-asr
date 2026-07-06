import importlib
import pkgutil

import livetrans


def test_all_livetrans_modules_are_importable():
    modules = [
        module.name
        for module in pkgutil.iter_modules(livetrans.__path__, prefix="livetrans.")
        if not module.ispkg
    ]

    assert modules
    for module_name in modules:
        importlib.import_module(module_name)
