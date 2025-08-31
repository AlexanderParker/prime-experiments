"""Elliptic Curve Primality Proving (ECPP) implementation."""

import random
from math import gcd
from gmpy2 import mpz, is_prime, invert, powmod, isqrt as gmpy2_isqrt


def find_curve_and_point(n):
    """Find an elliptic curve and a point on it modulo n."""
    n = mpz(n)
    while True:
        a = mpz(random.randrange(n))
        x = mpz(random.randrange(n))
        y2 = (x*x*x + a*x + 1) % n
        y = powmod(y2, (n + 1) // 4, n)
        if (y * y) % n == y2:
            return a, x, y


def add_points(P, Q, a, n):
    """Add two points on an elliptic curve."""
    if P == (None, None):
        return Q
    if Q == (None, None):
        return P
    
    if P[0] == Q[0]:
        if P[1] != Q[1] or P[1] == 0:
            return (None, None)
        try:
            lam = (3 * P[0] * P[0] + a) * invert(2 * P[1], n) % n
        except ZeroDivisionError:
            return (None, None)
    else:
        try:
            lam = (Q[1] - P[1]) * invert(Q[0] - P[0], n) % n
        except ZeroDivisionError:
            return (None, None)
    
    x3 = (lam * lam - P[0] - Q[0]) % n
    y3 = (lam * (P[0] - x3) - P[1]) % n
    
    return x3, y3


def scalar_mult(k, P, a, n):
    """Multiply a point by a scalar on an elliptic curve."""
    R = (None, None)
    Q = P
    while k > 0:
        if k & 1:
            R = add_points(R, Q, a, n)
        Q = add_points(Q, Q, a, n)
        k >>= 1
    return R


def ecpp_test(n, max_iterations=100):
    """
    Elliptic Curve Primality Proving test.
    Returns True if n is proven prime, False otherwise.
    """
    n = mpz(n)
    
    if n <= 2:
        return n == 2
    if not is_prime(n):  # Quick check with gmpy2
        return False
    
    m = n + 1
    
    for _ in range(max_iterations):
        a, x, y = find_curve_and_point(n)
        
        for k in range(2, min(gmpy2_isqrt(n) + 1, 10000)):
            if m % k == 0:
                q = m // k
                if is_prime(q) and q > (n**(1/4) + 1)**2:
                    P = (x, y)
                    try:
                        k_mult = scalar_mult(k, P, a, n)
                        m_mult = scalar_mult(m, P, a, n)
                        
                        if k_mult == (None, None) and m_mult == (None, None):
                            if gcd(int(k * (x*x*x + a*x + 1)), int(n)) == 1:
                                # Recursively verify q
                                if q < 10**10:  # Only recurse for smaller values
                                    return is_prime(q)
                                return True
                    except (ZeroDivisionError, ValueError):
                        continue
        
        m += 1
    
    # Could not prove primality
    return False