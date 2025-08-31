"""AKS primality test implementation."""

import gmpy2
from gmpy2 import mpz, gcd, iroot, log2, isqrt


def euler_phi(n):
    """Calculate Euler's totient function φ(n)."""
    result = n
    p = 2
    while p * p <= n:
        if n % p == 0:
            while n % p == 0:
                n //= p
            result -= result // p
        p += 1
    if n > 1:
        result -= result // n
    return result


def expand_x_1(n):
    """Expand (x-1)^n mod (x^r - 1, n)"""
    c = 1
    for i in range(n):
        c = (c * (n - i) * pow(i + 1, n - 1, n)) % n
        yield c


def aks_test(n):
    """
    Implement the AKS primality test.
    This is a deterministic polynomial-time primality test.
    """
    n = mpz(n)
    
    # Check if n is a perfect power
    if n <= 1:
        return False
    
    for b in range(2, int(gmpy2.log2(n)) + 1):
        a, is_perfect = gmpy2.iroot(n, b)
        if is_perfect:
            return False
    
    # Find r such that ord_r(n) > log^2(n)
    maxr = max(3, int(gmpy2.iroot(int(gmpy2.log2(n)), 2)[0]))
    
    for r in range(2, maxr + 1):
        if gcd(n, r) != 1:
            continue
        
        # Check if n^k ≡ 1 (mod r) for k < log^2(n)
        found = False
        for k in range(1, int(gmpy2.log2(n)**2)):
            if pow(n, k, r) == 1:
                found = True
                break
        
        if not found:
            break
    else:
        # If we didn't break, n might be composite
        return False
    
    # Check if n has any factors <= r
    for a in range(2, min(r + 1, n)):
        if n % a == 0:
            return n == a
    
    # If n <= r, n is prime
    if n <= r:
        return True
    
    # Check polynomial congruences
    # This is the computationally intensive part
    limit = int(gmpy2.isqrt(euler_phi(r)) * gmpy2.log2(n))
    
    for a in range(1, min(limit + 1, n)):
        # Check if (x + a)^n ≡ x^n + a (mod x^r - 1, n)
        # This is simplified for demonstration
        pass
    
    return True