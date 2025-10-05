"""
The utility module contains helper functions/classes that are not related
to internal functionalities and attributes of core modules. That means, the
functions/classes within this module can only be imported from other modules.

Instructions:
---
* The only provided function is ``to_level`` which normalizes the string of "sense"
within the PDTB dataset by reducing its sense level from a higher level to a lower
one. Usage is described in its docstring.
* Other than that, you're welcome to add any functionalities in this module
"""

from torch import Tensor
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_precision,
    multiclass_recall,
)


def to_level(sense: str, level: int = 2) -> str:
    """converts a sense in string to a desired level

    There are 3 sense levels in PDTB:
        Level 1 senses are the single-word senses like `Temporal` and `Contingency`.
        Level 2 senses add an additional sub-level sense on top of Level 1 senses, as in `Expansion.Exception`
        Level 3 senses adds yet another sub-level sense, as in `Temporal.Asynchronous.Precedence`.

    This function is used to ensure that all senses do not exceed the desired
    sense level provided as the argument `level`. For example,
    >>> to_level('Expansion.Restatement', level=1)
    'Expansion'
    >>> to_level('Temporal.Asynchronous.Succession', level=2)
    'Temporal.Asynchronous'

    When the input sense has a lower sense level than the desired sense level,
    this function will retain the original sense string. For example,

    >>> to_level('Expansion', level=2)
    'Expansion'
    >>> to_level('Comparison.Contrast', level=3)
    'Comparison.Contrast'

    Args:
        sense (str): a sense as given in any of the PDTB data files
        level (int): a desired sense level

    Returns:
        str: a sense below or at the desired sense level
    """
    s_split = sense.split(".")
    s_join = ".".join(s_split[:level])
    return s_join


sense_levels: dict[int, int] = {
    1: 5,
    2: 16,
    3: 21,
}

sense_map_1: dict[str, int] = {
    "Comparison": 0,
    "Contingency": 1,
    "EntRel": 2,
    "Expansion": 3,
    "Temporal": 4,
}


sense_map_2: dict[str, int] = {
    "Comparison": 0,
    "Comparison.Concession": 1,
    "Comparison.Contrast": 2,
    "Contingency": 3,
    "Contingency.Cause": 4,
    "Contingency.Condition": 5,
    "EntRel": 6,
    "Expansion": 7,
    "Expansion.Alternative": 8,
    "Expansion.Conjunction": 9,
    "Expansion.Exception": 10,
    "Expansion.Instantiation": 11,
    "Expansion.Restatement": 12,
    "Temporal": 13,
    "Temporal.Asynchronous": 14,
    "Temporal.Synchrony": 15,
}


sense_map_3: dict[str, int] = {
    "Comparison": 0,
    "Comparison.Concession": 1,
    "Comparison.Contrast": 2,
    "Contingency": 3,
    "Contingency.Cause": 4,
    "Contingency.Cause.Reason": 5,
    "Contingency.Cause.Result": 6,
    "Contingency.Condition": 7,
    "EntRel": 8,
    "Expansion": 9,
    "Expansion.Alternative": 10,
    "Expansion.Alternative.Chosen alternative": 11,
    "Expansion.Conjunction": 12,
    "Expansion.Exception": 13,
    "Expansion.Instantiation": 14,
    "Expansion.Restatement": 15,
    "Temporal": 16,
    "Temporal.Asynchronous": 17,
    "Temporal.Asynchronous.Precedence": 18,
    "Temporal.Asynchronous.Succession": 19,
    "Temporal.Synchrony": 20,
}


def map_senses(sense: str, level: int) -> int:
    sense = to_level(sense=sense, level=level)
    match level:
        case 1:
            return sense_map_1[sense]
        case 2:
            return sense_map_2[sense]
        case 3:
            return sense_map_3[sense]
        case _:
            raise ValueError(f"Only supports sense levels 1, 2, and 3. Got {level}")


def accuracy(pred: Tensor, target: Tensor) -> float:
    """Computes accuracy for given prediction and target tensors"""
    return multiclass_accuracy(pred, target).item()


def recall(pred: Tensor, target: Tensor, num_classes: int) -> float:
    """Computes recall for given prediction and target tensors"""
    return multiclass_recall(
        pred, target, average="macro", num_classes=num_classes
    ).item()


def precision(pred: Tensor, target: Tensor, num_classes: int) -> float:
    """Computes precision for given prediction and target tensors"""
    return multiclass_precision(
        pred, target, average="macro", num_classes=num_classes
    ).item()
