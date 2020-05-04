from typing import Sequence


def smooth(values: Sequence, factor: float):
    result = [values[0]]
    for v in values[1:]:
        result.append(result[-1] * factor + (1. - factor) * v)
    return result
