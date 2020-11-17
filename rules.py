from src.fuzzyevaluator.evaluate import FuzzySystem, LarsenFuzzySystem, MamdaniFuzzySystem
from src.fuzzydsl.dsl import Parser

PROGRAM = """
Temperature 10 40 100 : Cold, Medium, Hot
Humidity 20 100 100 : Dry, Normal, Wet
Speed 0 100 100 : Slow, Moderate, Fast

= Cold TriangularSet 10 25 10
= Medium TriangularSet 15 35 25
= Hot TriangularSet 25 40 40

= Dry TriangularSet 60 100 100
= Wet TriangularSet 20 60 20
= Normal TrapezoidSet 30 50 70 90

= Slow TriangularSet 0 50 0
= Moderate TriangularSet 10 90 50
= Fast TriangularSet 50 100 100

if Temperature is Cold and Humidity is Wet then Speed is Slow
if Temperature is Cold and Humidity is Normal then Speed is Slow
if Temperature is Medium and Humidity is Wet then Speed is Slow
if Temperature is Medium and Humidity is Normal then Speed is Moderate
if Temperature is Cold and Humidity is Dry then Speed is Moderate
if Temperature is Hot and Humidity is Wet then Speed is Moderate
if Temperature is Hot and Humidity is Normal then Speed is Fast
if Temperature is Hot and Humidity is Dry then Speed is Fast
if Temperature is Medium and Humidity is Dry then Speed is Fast
"""

PROGRAM2 = """
x1 0 100 100 : S, M, L
x2 0 100 100 : S, M, L
y  0 100 100 : S, M, L
z  0 100 100 : S, M, L

= S TriangularSet 0 50 25
= M TriangularSet 25 75 50
= L TriangularSet 50 100 75

if x1 is S and x2 is M then y is S
if x1 is S and x2 is M then z is L

if x1 is M and x2 is M then y is M
if x1 is M and x2 is M then z is M

if x1 is L and x2 is L then y is L
if x1 is L and x2 is L then z is S

if x1 is S and x2 is M then y is S
if x1 is S and x2 is M then z is L

if x1 is M and x2 is S then y is S
if x1 is M and x2 is S then z is L

if x1 is L and x2 is M then y is L
if x1 is L and x2 is M then z is S

if x1 is M and x2 is L then y is L
if x1 is M and x2 is L then z is S

if x1 is L and x2 is S then y is M
if x1 is L and x2 is S then z is M

if x1 is S and x2 is L then y is M
if x1 is S and x2 is L then z is M
"""

def _main():
    parser = Parser()
    sets, rules = parser(PROGRAM)

    system = MamdaniFuzzySystem(sets, rules)
    print(system.evaluate({"Temperature": 18, "Humidity": 60}))

    system2 = LarsenFuzzySystem(sets, rules )
    print(system2.evaluate({"Temperature": 18, "Humidity": 60}))

    sets2, rules2 = parser(PROGRAM2)
    
    system3 = MamdaniFuzzySystem(sets2, rules2)
    print(system3.evaluate({'x1': 35, 'x2': 75}))

if __name__ == "__main__":
    _main()