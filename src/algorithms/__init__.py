"""Prime number algorithms module."""

from .primality import is_prime, next_prime
from .mersenne import MersennePrime
from .aks import aks_test
from .ecpp import ecpp_test

__all__ = [
    'is_prime',
    'next_prime',
    'MersennePrime',
    'aks_test',
    'ecpp_test'
]