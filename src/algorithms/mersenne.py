"""Mersenne prime calculations and analysis."""

import gmpy2
from gmpy2 import mpz, log10, sqrt, is_prime, is_even
from typing import List, Set, Tuple


class MersennePrime:
    """Class for working with Mersenne primes."""
    
    def __init__(self, exponent: int):
        """Initialize with Mersenne exponent."""
        self.exponent = mpz(exponent)
        self.value = mpz(2) ** self.exponent - 1
        self.digits = self.estimate_digits(self.value)
    
    @staticmethod
    def estimate_digits(n):
        """Estimate number of digits in a large number."""
        return int(gmpy2.log10(n)) + 1
    
    def get_first_n_digits(self, n: int) -> str:
        """Get the first n digits of the Mersenne prime."""
        return str(self.value // (mpz(10) ** (self.digits - n)))[:n]
    
    def get_last_n_digits(self, n: int) -> str:
        """Get the last n digits of the Mersenne prime."""
        return str(self.value % (mpz(10) ** n)).zfill(n)
    
    def calculate_modular_remainders(self, max_prime: int = None) -> Set[int]:
        """
        Calculate modular remainders for prime moduli up to max_prime.
        If max_prime is None, uses sqrt(mersenne_prime) as the limit.
        """
        if max_prime is None:
            max_prime = int(sqrt(self.value) + 1)
        
        mods = set()
        checked = 0
        
        for i in range(2, max_prime):
            if is_prime(i):
                mod = -(self.value % -i)
                if is_even(mod):  # Prime gaps are never odd (except 2 to 3)
                    mods.add(mod)
                checked += 1
                
                if checked % 1000 == 0:
                    print(f"Checked {checked} prime moduli...", end='\r')
        
        print(f"\nCompleted: {checked} prime moduli checked, {len(mods)} unique mods found")
        return mods
    
    def find_missing_evens(self, even_list: List[int]) -> List[int]:
        """Find missing even numbers in a sorted list of evens."""
        missing = []
        for i in range(len(even_list) - 1):
            current = even_list[i]
            next_num = even_list[i + 1]
            for num in range(current + 2, next_num, 2):
                missing.append(num)
        return missing
    
    def check_next_primes(self, offsets: List[int]) -> List[Tuple[int, bool]]:
        """
        Check if mersenne_prime + offset is prime for each offset.
        Returns list of (offset, is_prime) tuples.
        """
        results = []
        for offset in offsets:
            candidate = self.value + offset
            results.append((offset, is_prime(candidate)))
        return results
    
    def __str__(self):
        return f"M{self.exponent} (2^{self.exponent} - 1, {self.digits} digits)"
    
    def __repr__(self):
        return f"MersennePrime({self.exponent})"


# Known Mersenne prime exponents
KNOWN_MERSENNE_EXPONENTS = [
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281,
    3217, 4253, 4423, 9689, 9941, 11213, 19937, 21701, 23209, 44497, 86243,
    110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221,
    3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457,
    32582657, 37156667, 42643801, 43112609, 57885161, 74207281, 77232917,
    82589933  # 51st known Mersenne prime
]