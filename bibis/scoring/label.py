Label = int | float | str
NEGATIVE_LABEL, POSITIVE_LABEL = 0, 1

NO_LABEL = "__NO_LABEL__"

def str2label(s: str) -> Label:
    if s == NO_LABEL:
        return s
    try:
        l = int(s)
    except ValueError:
        l = float(s)
    return l