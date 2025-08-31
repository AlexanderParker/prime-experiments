"""Prime counting and estimation functions."""

import gmpy2
from gmpy2 import mpz, log, mpfr
import math


def estimate_primes_count(n):
    """
    Estimate the number of primes up to n using the Prime Number Theorem.
    π(n) ≈ n / ln(n)
    """
    n = mpz(n)
    if n < 2:
        return mpz(0)
    return gmpy2.floor(mpfr(n) / log(n))


def estimate_primes_count_improved(n):
    """
    Improved estimate using the logarithmic integral.
    Li(n) provides a better approximation than n/ln(n).
    """
    n = mpz(n)
    if n < 2:
        return mpz(0)
    
    # Simplified logarithmic integral approximation
    # Li(n) ≈ n/ln(n) * (1 + 1/ln(n) + 2/ln²(n) + ...)
    ln_n = float(log(n))
    if ln_n <= 0:
        return mpz(0)
    
    # Using first few terms of the asymptotic expansion
    estimate = float(n) / ln_n
    estimate *= (1 + 1/ln_n + 2/(ln_n**2) + 6/(ln_n**3))
    
    return mpz(int(estimate))