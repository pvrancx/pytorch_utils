from typing import Sequence


def linear_annealing(min_val:float, max_val: float, fraction:float) -> float:
    return min_val + (max_val - min_val) * fraction


def exponential_annealing(min_val:float, max_val: float, fraction:float) -> float:
    return min_val * (max_val / min_val) ** fraction


def smooth(values: Sequence, factor: float) -> Sequence[float]:
    result = [values[0]]
    for v in values[1:]:
        result.append(result[-1] * factor + (1. - factor) * v)
    return result


