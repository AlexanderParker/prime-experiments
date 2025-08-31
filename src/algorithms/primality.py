"""Basic primality testing algorithms."""

import gmpy2
from gmpy2 import mpz, is_prime as gmpy2_is_prime, next_prime as gmpy2_next_prime


def is_prime(x):
    """
    Check if a number is prime using gmpy2 for large numbers,
    falling back to trial division for small numbers.
    """
    if isinstance(x, int) and x < 10**6:
        return is_prime_trial_division(x)
    return gmpy2_is_prime(mpz(x))


def is_prime_trial_division(x):
    """Basic trial division primality test for small numbers."""
    if x < 2:
        return False
    if x == 2:
        return True
    if x % 2 == 0:
        return False
    for i in range(3, int(x**0.5) + 1, 2):
        if x % i == 0:
            return False
    return True


def next_prime(n):
    """Find the next prime number after n."""
    if isinstance(n, int) and n < 10**6:
        return next_prime_basic(n)
    return int(gmpy2_next_prime(mpz(n)))


def next_prime_basic(n):
    """Find next prime using trial division."""
    next_num = n + 1
    while not is_prime_trial_division(next_num):
        next_num += 1
    return next_num


def next_prime_mod(numbers):
    """
    Find the next number that is not divisible by any of the given prime numbers.
    """
    last_number = max(numbers)
    for step in range(last_number + 1, last_number * 2):
        if all(step % number != 0 for number in numbers):
            return step
    return None