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