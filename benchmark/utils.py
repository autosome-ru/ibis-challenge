from functools import singledispatchmethod, update_wrapper

def _register(self, cls, method=None):
    if hasattr(cls, "__func__"):
        setattr(cls, "__annotations__", cls.__func__.__annotations__)
    return self.dispatcher.register(cls, func=method)

singledispatchmethod.register = _register

END_LINE_CHARS = "\r\n"

GLOBAL_CONVERTORS = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool}

def register_global_converter(type_name, converter):
    global GLOBAL_CONVERTORS
    GLOBAL_CONVERTORS[type_name] = converter


def auto_convert(cls, fields):
    results = []
    for field in fields:
        if field.converter is not None:
            results.append(field)
            continue
        converter = GLOBAL_CONVERTORS.get(field.type, None)
        results.append(field.evolve(converter=converter))
    return results