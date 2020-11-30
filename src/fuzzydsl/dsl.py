from __future__ import annotations
from enum import Enum
import re
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

# Tokenizing Region


class TokenType(Enum):
    """
    Declare a token type for each terminal of the grammar.
    """

    If = 0
    Then = 1
    Is = 2
    Identifier = 3
    Space = 4
    And = 5
    Or = 6
    Colon = 7
    Comma = 8
    Function = 9
    Num = 10
    EqualSign = 11
    Not = 12


class Token(object):
    """
    Class that represent a token after tokenization. Each token maps to a terminal
    in the DSL grammar.
    """

    def __init__(self, tokenType: TokenType, lexeme: str) -> None:
        self.tokenType = tokenType
        self.lexeme = lexeme

    def __str__(self) -> str:
        return str(self.tokenType)

    def __repr__(self) -> str:
        return str(self)


tokenTable = [
    (r"if", TokenType.If),
    (r"then", TokenType.Then),
    (r"is", TokenType.Is),
    (r"and", TokenType.And),
    (r"or", TokenType.Or),
    (r"not", TokenType.Not),
    (r":", TokenType.Colon),
    (r",", TokenType.Comma),
    (r"(TriangularSet)|(TrapezoidSet)", TokenType.Function),
    (r"=", TokenType.EqualSign),
    (r"\d+", TokenType.Num),
    (r"\s", TokenType.Space),
    (r"\w+", TokenType.Identifier),
]


def tokenize(text: str) -> List[Token]:
    """
    Split input string into Tokens, scanning each regex to the start of the string
    trying to find the first match (according to priority in tokens).
    """
    tokens = []
    while text:
        for regex, tokenType in tokenTable:
            # Search in tokenTable in priority order and
            # process the first match
            match = re.match(regex, text)
            # Ignore blank tokens, allowing any amount of
            # spaces, newlines, etc.
            if match and tokenType != TokenType.Space:
                lexeme = match.group()
                # Found a match, append it to token list
                text = text[len(lexeme) :]
                tokens.append(Token(tokenType, lexeme))
                break
            elif match and tokenType == TokenType.Space:
                text = text[len(match.group()) :]
                break
    return tokens


# Returns first element of xs. Just a functional way of
# first index a list, just because i love functional programming.
def first(xs: List[Token]):
    return xs[0]


# AST region


class FuzzyVar:
    """
    Represents a fuzzy variable.
    """

    def __init__(self, name: str, min=0, max=1, resolution=10) -> None:
        self.name = name
        self.min = min
        self.max = max
        self.resolution = resolution

        self.domain = np.linspace(min, max, resolution)

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, o: FuzzyVar) -> bool:
        return self.name == o.name


class Formula:
    """
    Represents a fuzzy formula formed by conjuction of fuzzy statements.
    """

    def __init__(self, *args: Statement) -> None:
        self.vars = args

    def __add__(self, other):
        assert isinstance(other, Formula)
        return Formula(*(self.Statements + other.Statements))

    def __str__(self) -> str:
        conditions = self.Statements
        if len(conditions) == 1:
            result = (
                f"{conditions[0].subject.name} is {conditions[0].classification.name}"
            )
            if conditions[0].neg:
                return f"not {result}"
            else:
                return result
        else:
            result = ""
            for cond in conditions:
                if cond.neg:
                    result += "not "
                result += f"{cond.subject.name} is {cond.classification.name} AND "
            return result[: len(result) - 5]

    @property
    def Statements(self) -> Tuple[Statement, ...]:
        return self.vars

    def Fired(self, *xs: float) -> float:
        assert len(xs) == len(self.Statements)
        activationValue = min(stm.Fired(x) for stm, x in zip(self.Statements, xs))
        return activationValue

    def validInput(self, *vars: FuzzyVar):
        return all(x.subject in vars for x in self.Statements)

    @property
    def Names(self):
        return tuple(x.subject for x in self.Statements)


class FuzzyClassification:
    """
    Rerpresents a fuzzy set or fuzzy classification.
    """

    def __init__(self, name: str, low=0, high=1, resolution=10) -> None:
        self.name = name
        self.low = low
        self.high = high
        self.resolution = resolution

        self.domain = np.linspace(low, high, resolution)

        self._membership_func: Optional[Callable] = None
        self._type = ""

    def __eq__(self, o: FuzzyClassification) -> bool:
        return self.name == o.name

    def __repr__(self) -> str:
        return f" is {self.name}"

    def __hash__(self) -> int:
        return hash(self.name)

    def __call__(self, x: float) -> float:
        return self._membership_func(x)

    def _stretchToDomain(self, val):
        """
        Returns the element from domain with the least
        distance from val.
        """
        # Calcular la distancia entre cada elemento y val
        distanceVector = np.abs(self.domain - val)
        # devolver el elemento del dominio con la menor distancia
        minDistance = np.argmin(distanceVector)
        return self.domain[minDistance]

    def _becomeTriangularSet(self, _min, _max, middle):
        self._type = "Triangular"
        _min = self._stretchToDomain(_min)
        _max = self._stretchToDomain(_max)
        middle = self._stretchToDomain(middle)

        if _min == middle:
            cachedDomain = np.round(
                np.maximum((_max - self.domain) / (_max - middle), 0), 2
            )

            self._membership_func = lambda x: cachedDomain[
                np.abs(self.domain - x).argmin()
            ]

        elif middle == _max:
            cachedDomain = np.round(
                np.maximum((self.domain - _min) / (middle - _min), 0), 2
            )

            self._membership_func = lambda x: cachedDomain[
                np.abs(self.domain - x).argmin()
            ]

        else:
            cachedDomain = np.round(
                np.maximum(
                    np.minimum(
                        (self.domain - _min) / (middle - _min),
                        (_max - self.domain) / (_max - middle),
                    ),
                    0,
                ),
                2,
            )

            self._membership_func = lambda x: cachedDomain[
                np.abs(self.domain - x).argmin()
            ]

    def _becomeTrapezoidSet(self, a, b, c, d):
        self._type = "Trapezoid"
        a = self._stretchToDomain(a)
        b = self._stretchToDomain(b)
        c = self._stretchToDomain(c)
        d = self._stretchToDomain(d)

        cachedDomain = np.round(
            np.minimum(
                np.maximum(
                    np.minimum(
                        (self.domain - a) / (b - a), (d - self.domain) / (d - c)
                    ),
                    0,
                ),
                1,
            ),
            2,
        )

        self._membership_func = lambda x: cachedDomain[np.abs(self.domain - x).argmin()]

    @property
    def Type(self):
        return self._type


class Statement:
    """
    Represents a statement in the form: x is A.
    """

    def __init__(
        self, subject: FuzzyVar, classification: FuzzyClassification, neg: bool = False
    ) -> None:
        self.subject = subject
        self.classification = classification
        self.neg = neg

    def Fired(self, x) -> float:
        return 1 - self.classification(x) if self.neg else self.classification(x)

    def __repr__(self) -> str:
        return f"{self.subject.name} is {self.classification.name}"


class Rule:
    """
    Represents a rule or an implication.
    """

    def __init__(self, condition: Formula, implication: Statement) -> None:
        self.condition = condition
        self.implication = implication

    def Fired(self, *xs: float):
        return self.condition.Fired(*xs) > 0

    def apply(self, *xs: float) -> Optional[float]:
        val = self.condition.Fired(*xs)
        if val > 0:
            return val
        return None

    def applyLarsen(self, *xs: float):
        val = self.condition.Fired(*xs)
        if val > 0:
            domain = [
                self.implication.classification(x)
                for x in self.implication.classification.domain
            ]
            return np.multiply(domain, val)
        return None

    def canApply(self, *vars: FuzzyVar):
        return self.condition.validInput(*vars)

    @property
    def Vars(self):
        return self.condition.Names

    def __str__(self) -> str:
        return f"{self.condition} --> {self.implication}"

    def __repr__(self) -> str:
        return str(self)


# PARSER region


class Parser:
    """
    LL parser that returns uppon call, a list of rules and
    the fuzzy sets and available classifications for each
    variable available.
    """

    def __init__(self) -> None:
        self.tokens: List[Token] = []
        self.rules: List[Rule] = []
        self._sets: Dict[str, FuzzyClassification] = {}
        self.fuzzySets: Dict[FuzzyVar, List[FuzzyClassification]] = {}

    def __call__(self, text: str):
        # Split input string in tokens and update
        # current parser state.
        self.tokens = tokenize(text)
        # Parse text to obtain program structure.
        self._parseText()
        return self.fuzzySets, self.rules

    def _consume(self, tokenType: TokenType):
        """
        Try to consume a token of the desired type from the
        token list and return that token.
        """
        tokens = self.tokens
        # Current token must be of the input Type
        assert first(tokens).tokenType == tokenType
        token = first(tokens)
        # Consume 1 token
        self.tokens = self.tokens[1:]
        return token

    def _parseRule(self):
        """
        Function that implement Rule non-terminal in the grammar.
        """
        # Recognize the 'if' keyword
        self._consume(TokenType.If)
        # Next comes a Formula
        left = self._parseFormula()
        # Next need to match a 'then' keyword
        self._consume(TokenType.Then)
        # Lastly comes another Formula
        right = self._parseStatement()
        return Rule(left, right)

    def _parseStatement(self, neg=False):
        """
        Fuction that parses the Statement non-Terminal.
        """
        subject = FuzzyVar("")
        if self.tokens[0].tokenType == TokenType.Not:
            neg = not neg
            self._consume(TokenType.Not)
            return self._parseStatement(neg)
        # First comes an identifier that is a variable
        id_ = self._consume(TokenType.Identifier)
        # Then comens the 'is' keyword
        for key in self.fuzzySets:
            if id_.lexeme == key.name:
                subject = key
                break
        self._consume(TokenType.Is)
        # Now comes the classification (annother identifier)
        classification = FuzzyClassification(self._consume(TokenType.Identifier).lexeme)
        for slist in self.fuzzySets.values():
            for fuzzySet in slist:
                if fuzzySet.name == classification.name:
                    classification = fuzzySet
                    break
        # Check that classification is in subject's available classifications
        assert (
            classification in self.fuzzySets[subject]
        ), f"{classification.name} is not part of {subject.name} classifications"

        # Construct and return the statement
        return Statement(subject, classification, neg)

    def _parseFormula(self, neg=False):
        """
        Function that parses the Formula non-Terminal.
        """
        # Match an Statement
        form = self._parseStatement()
        form.neg = neg

        # If 'and' keyword comes next, then match another formula
        if self.tokens and self.tokens[0].tokenType == TokenType.And:
            self._consume(TokenType.And)
            return Formula(form) + self._parseFormula()
        elif self.tokens and self.tokens[0].tokenType == TokenType.Or:
            self._consume(TokenType.Or)
            form.neg = True
            return Formula(form) + self._parseFormula(neg=True)
        else:
            return Formula(form)

    def _parseFuzzySet(self, l, h, r):
        classification = self._consume(TokenType.Identifier)
        try:
            fuzzyClass = self._sets[classification.lexeme]
        except KeyError:
            fuzzyClass = FuzzyClassification(classification.lexeme, l, h, r)
            self._sets[classification.lexeme] = fuzzyClass
        if first(self.tokens).tokenType != TokenType.Comma:
            return [fuzzyClass]
        else:
            self._consume(TokenType.Comma)
            return [fuzzyClass] + self._parseFuzzySet(l, h, r)

    def _parseSetDescription(self):
        # Consume the equal sign
        self._consume(TokenType.EqualSign)
        # Should come an identifier representing a FuzzySet
        fuzzySetName = self._consume(TokenType.Identifier).lexeme
        assert fuzzySetName in self._sets
        # Next comes a function description
        func = self._consume(TokenType.Function).lexeme
        args = []

        # Next should come a list of integers representing function's args
        while self.tokens and first(self.tokens).tokenType == TokenType.Num:
            arg = int(self._consume(TokenType.Num).lexeme)
            args.append(arg)

        return fuzzySetName, func, args

    def _parseText(self):
        assert first(self.tokens).tokenType == TokenType.Identifier, "Define Sets first"
        # Parse the sets definitions
        while first(self.tokens).tokenType == TokenType.Identifier:
            next_input = self._consume(TokenType.Identifier).lexeme

            # Now should come 3 integers describing this variable (min max resolution)
            min_ = int(self._consume(TokenType.Num).lexeme)
            max_ = int(self._consume(TokenType.Num).lexeme)
            res = int(self._consume(TokenType.Num).lexeme)

            self._consume(TokenType.Colon)
            # Create every Fuzzy Set for this variable, using this range domain
            sets = self._parseFuzzySet(min_, max_, res)

            input_ = FuzzyVar(next_input, min_, max_, res)
            self.fuzzySets[input_] = sets

        # Parse fuzzy Sets Conformation
        while self.tokens and first(self.tokens).tokenType == TokenType.EqualSign:
            name, func, args = self._parseSetDescription()
            for vlist in self.fuzzySets.values():
                for fuzzySet in vlist:
                    if name == fuzzySet.name:
                        if func == "TrapezoidSet":
                            fuzzySet._becomeTrapezoidSet(*args)
                        elif func == "TriangularSet":
                            fuzzySet._becomeTriangularSet(*args)

        # Parse the rules
        while self.tokens and first(self.tokens).tokenType == TokenType.If:
            self.rules.append(self._parseRule())