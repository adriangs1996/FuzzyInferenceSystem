from typing import Callable, Dict, Iterable, List
from ..fuzzydsl.dsl import FuzzyClassification, FuzzyVar, Rule
import numpy as np
from bisect import bisect_left


def minimum(*xs: Iterable[float]):
    """
    Parameters: xs, iterables of the same length.
    Output: list with the least element of every position of each iterable.
    """
    return [np.min(list(x)) for x in zip(*xs)]


def maximum(*xs: Iterable[float]):
    return [np.max(list(x)) for x in zip(*xs)]


# Desfuzzification methods
def centroid(var: FuzzyVar, firedRules: Dict[FuzzyVar, List[float]]):
    numerator = np.sum(np.multiply(var.domain, firedRules[var]))
    denominator = np.sum(firedRules[var])
    return numerator / denominator


def weightedAverage(var: FuzzyVar, firedRules: Dict[FuzzyVar, List[float]]):
    pass


def maxPrinciple(var: FuzzyVar, firedRules):
    return var.domain[np.argmax(firedRules[var])]


def meanOfMaximum(var: FuzzyVar, firedRules):
    min_ = np.max(firedRules[var])
    numerator = np.sum(
        [var.domain[i] for i, x in enumerate(firedRules[var]) if x == min_]
    )
    denominator = len([x for x in firedRules[var] if x == min_])
    return numerator / denominator


def bisector(var: FuzzyVar, firedRules):
    # Find area and shrink it by 2
    searchedArea = np.sum(firedRules[var]) / 2
    # Find the index where area match half the total area.
    # Could use a for loop fot this but mehhh.... i love
    # functional way. It is just a sum array and then a
    # binary search.
    array = list(
        map(lambda i: sum(firedRules[var][:i]), range(1, len(firedRules[var]) + 1))
    )
    return var.domain[bisect_left(array, searchedArea)]


methodTable = {
    "maximum": maxPrinciple,
    "centroid": centroid,
    "weightedAverage": weightedAverage,
    "mom": meanOfMaximum,
    "bisector": bisector,
}


class FuzzySystem:
    def __init__(
        self, sets: Dict[FuzzyVar, List[FuzzyClassification]], rules: List[Rule]
    ):
        self.sets = sets
        self.rules = rules
        self._variables = list(sets.keys())
        self._fuzzySets = []

        for l in sets.values():
            self._fuzzySets + l

        self._fuzzySets = list(set(self._fuzzySets))

    def evaluate(self, inputVariables: Dict[str, float]):
        """
        This method should be implemented by inheritors to
        make the system evaluate a given set of inputs.
        """
        raise NotImplementedError


class MamdaniFuzzySystem(FuzzySystem):
    """
    Provides an integrated view over a Fuzzy System and allow operations
    over it. It can be evaluated with input variables to retrieve result.
    """

    def __init__(
        self,
        sets: Dict[FuzzyVar, List[FuzzyClassification]],
        rules: List[Rule],
        method: str = "centroid",
    ):
        super().__init__(sets, rules)
        self._desfuzzify: Callable = methodTable[method]

    def evaluate_rules(self, input_: Dict[FuzzyVar, float]):
        firedRules = {}
        for rule in self.rules:
            # A Rule can only be evaluated if all variables are the present
            if rule.canApply(*input_.keys()):
                inputs = tuple(input_[var] for var in rule.Vars)
                # Calculate the antecedents contribution
                contribution = rule.apply(*inputs)
                if contribution is not None:
                    # Modify output Fuzzy Set based on antencedent contribution
                    domain = [
                        rule.implication.classification(x)
                        for x in rule.implication.classification.domain
                    ]
                    truncatedSet = np.minimum(domain, contribution)
                    try:
                        firedRules[rule.implication.subject].append(truncatedSet)
                    except KeyError:
                        firedRules[rule.implication.subject] = [truncatedSet]
        return firedRules

    def aggregate_rules(self, firedRules):
        # Agregate every output var's truncated domain
        for var in firedRules.keys():
            firedRules[var] = maximum(*firedRules[var])

    def desfuzzify(self, firedRules):
        # Desfuzzify every variable's current Fuzzy Set
        for var in firedRules.keys():
            firedRules[var] = self._desfuzzify(var, firedRules)

    def evaluate(self, inputVariables: Dict[str, float]):
        # Search for rules that fires with variables's input
        input_: Dict[FuzzyVar, float] = {
            key: inputVariables[key.name]
            for key in self.sets.keys()
            if key.name in inputVariables
        }
        firedRules = self.evaluate_rules(input_)

        self.aggregate_rules(firedRules)

        self.desfuzzify(firedRules)

        return firedRules


class LarsenFuzzySystem(MamdaniFuzzySystem):
    def evaluate_rules(self, input_: Dict[FuzzyVar, float]):
        firedRules = {}
        for rule in self.rules:
            # A Rule can only be evaluated if all variables are the present
            if rule.canApply(*input_.keys()):
                inputs = tuple(input_[var] for var in rule.Vars)
                # Calculate the antecedents contribution
                strenght = rule.applyLarsen(*inputs)
                if strenght is not None:
                    try:
                        firedRules[rule.implication.subject].append(strenght)
                    except KeyError:
                        firedRules[rule.implication.subject] = [strenght]
        return firedRules