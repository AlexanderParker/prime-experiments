"""Finite certificate checks quoted in range_proof.md. Exact integer arithmetic only."""
from math import isqrt

def is_prime_td(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = isqrt(n)
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True

def primorial(n: int) -> int:
    out = 1
    for p in range(2, n + 1):
        if is_prime_td(p):
            out *= p
    return out

# primorials used
for q in (7, 11, 13, 23, 29, 31, 41):
    print(f"{q}# = {primorial(q)}")

# base witnesses
print("29,31 prime:", is_prime_td(29), is_prime_td(31))
print("59,61 prime:", is_prime_td(59), is_prime_td(61))
print("41,43 prime:", is_prime_td(41), is_prime_td(43))
print("149,151 prime:", is_prime_td(149), is_prime_td(151))

# E9 certificate: c_198437
g = 198437
print("198437 prime:", is_prime_td(g))
print("198437^2 mod 30 =", (g * g) % 30)
lo, hi = g * g + 10, g * g + 12
print("legs:", lo, hi)
print("legs prime (trial division to sqrt):", is_prime_td(lo), is_prime_td(hi))
print("hi <= 31#:", hi <= primorial(31))
print("39377242978 < lo:", 39377242978 < lo)

# no c_g revealed for 17 <= g <= 73
def missed_legs(g):
    r = (g * g) % 30
    if r == 1:
        return g * g + 28, g * g + 30
    if r == 19:
        return g * g + 10, g * g + 12
    raise ValueError
for g in range(7, 80):
    if is_prime_td(g):
        a, b = missed_legs(g)
        print(f"c_{g} = ({a}, {b}) revealed: {is_prime_td(a) and is_prime_td(b)}")

# 41 is a twin node not congruent 29 mod 30; twin nodes below 200 and their class mod 30
nodes = [p for p in range(29, 200) if is_prime_td(p) and is_prime_td(p + 2)]
print("twin nodes 29..199:", nodes)
print("classes mod 30:", sorted(set(p % 30 for p in nodes)))

# Good s at small twin nodes (s+ + 2 <= s#), s <= 200
def next_node(s):
    t = s + 1
    while not (is_prime_td(t) and is_prime_td(t + 2)):
        t += 1
    return t
for s in nodes[:6]:
    t = next_node(s)
    print(f"Good {s}: next node {t}, {t}+2 <= {s}# : {t + 2 <= primorial(s)}")
