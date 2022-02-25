from enum import Enum
from functools import singledispatchmethod, update_wrapper
from pathlib import Path
from exceptions import *

def _register(self, cls, method=None):
    if hasattr(cls, "__func__"):
        setattr(cls, "__annotations__", cls.__func__.__annotations__)
    return self.dispatcher.register(cls, func=method)

singledispatchmethod.register = _register

END_LINE_CHARS = "\r\n"

def safe_path(s: str) -> Path:
    path = Path(s)
    if not path.exists():
        msg = f"Path '{s}' doesn't exist"
        raise WrongPathException(msg)
    return path

GLOBAL_CONVERTERS = {
    "int": int,
    int: int,
    "float": float,
    float: float,
    "str": str,
    str: str,
    "bool": bool,
    bool: bool,
    "Path": safe_path,
    Path: safe_path}

def register_global_converter(type_, converter):
    global GLOBAL_CONVERTERS
    GLOBAL_CONVERTERS[type_] = converter
    GLOBAL_CONVERTERS[type_.__name__] = converter

def register_enum(enum_tp):
    def converter(x: str):
        x_canonical = x.upper()
        try:
            return enum_tp[x_canonical]
        except KeyError:
            pass
        msg = f"Wrong value '{x}' for enum '{enum_tp}'"
        possible_exc = "Wrong{enum_tp.__name__}Exception"
        if possible_exc in globals():
            globals()[possible_exc](msg)
        else:
            raise BenchmarkException(msg)
            
    register_global_converter(enum_tp, converter)
    return enum_tp 

def register_type(type_):
    register_global_converter(type_, lambda x: type_(x))
    return type_


def auto_convert(cls, fields):
    results = []
    for field in fields:

        if field.converter is not None:
            results.append(field)
            continue
        converter = GLOBAL_CONVERTERS.get(field.type, None)
        results.append(field.evolve(converter=converter))
    return results


def undict(src, dest=None, pref=None):
    if dest is None:
        dest = {}

    if isinstance(src, list):
        for ind, el in enumerate(src, 1):
            if pref is None:
                name = f"{ind}"
            else:
                name = f"{pref}_{ind}"
            if isinstance(el, list):
                undict(el, dest, pref=name)
            elif isinstance(el, dict):
                undict(el, dest, pref=name)
            else:
                dest[name] = el
    elif isinstance(src, dict):
        for key, value in src.items():
            if pref is None:
                name = f"{key}"
            else:
                name = f"{pref}_{key}"
            if isinstance(value, list):
                undict(value, dest, pref=name)
            elif isinstance(value, dict):
                undict(value, dest, pref=name)
            else:
                dest[name] = value
    return dest

if __name__ == "__main__":
    print(undict({"1": 50, 2: [2, 4, 5], "key": {5: 7, 4: {5:4}, 10: [1,2,3] }}))