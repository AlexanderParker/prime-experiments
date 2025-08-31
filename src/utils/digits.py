"""Utility functions for working with large number digits."""

import math
import gmpy2
from gmpy2 import mpz, log10


def estimate_digits(n):
    """Estimate the number of digits in a large number."""
    if isinstance(n, (int, float)) and n > 0:
        return math.floor(math.log10(n)) + 1
    elif isinstance(n, mpz) and n > 0:
        return int(gmpy2.log10(n)) + 1
    else:
        return 0


def get_first_n_digits(number, n):
    """Get the first n digits of a large number without converting entire number to string."""
    digits = estimate_digits(number)
    if digits <= n:
        return str(number)
    
    if isinstance(number, int):
        number = mpz(number)
    
    # Get first n digits by dividing by appropriate power of 10
    divisor = mpz(10) ** (digits - n)
    first_part = number // divisor
    return str(first_part)


def get_last_n_digits(number, n):
    """Get the last n digits of a large number without converting entire number to string."""
    if isinstance(number, int):
        number = mpz(number)
    
    return str(number % (mpz(10) ** n)).zfill(n)