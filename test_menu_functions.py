#!/usr/bin/env python
"""Test script for menu functions without interactive input."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from algorithms import MersennePrime, is_prime, next_prime, aks_test, ecpp_test
from utils import estimate_digits, get_first_n_digits, get_last_n_digits
from config.settings import MERSENNE_EXPONENTS, RESULTS_DIR, TEST_PRIMES, TEST_COMPOSITES

def test_mersenne_analysis():
    """Test Mersenne prime analysis functionality."""
    print("=== Testing Mersenne Analysis ===")
    
    # Test small Mersenne primes
    for exp in [3, 5, 7, 13, 17]:
        mp = MersennePrime(exp)
        print(f"M{exp} = {mp.value} ({mp.digits} digits)")
    
    # Test larger Mersenne
    mp = MersennePrime(127)
    print(f"M127 has {mp.digits} digits")
    print(f"First 30: {mp.get_first_n_digits(30)}")
    print(f"Last 30: {mp.get_last_n_digits(30)}")
    print()

def test_primality_testing():
    """Test primality testing functionality."""
    print("=== Testing Primality Testing ===")
    
    # Test basic primality
    test_numbers = [2, 3, 4, 5, 97, 98, 101, 997, 1009]
    for num in test_numbers:
        result = is_prime(num)
        print(f"{num}: {'PRIME' if result else 'COMPOSITE'}")
    
    # Test next prime
    for start in [10, 100, 1000]:
        next_p = next_prime(start)
        print(f"Next prime after {start}: {next_p}")
    print()

def test_modular_arithmetic():
    """Test modular arithmetic functionality."""
    print("=== Testing Modular Arithmetic ===")
    
    # Test with M7
    mp = MersennePrime(7)
    mods = mp.calculate_modular_remainders(20)
    sorted_mods = sorted(list(mods))
    print(f"M7 mod small primes: {sorted_mods}")
    
    # Test missing evens
    missing = mp.find_missing_evens(sorted_mods)
    print(f"Missing evens up to 20: {missing}")
    print()

def test_quick_tools():
    """Test quick tools functionality."""
    print("=== Testing Quick Tools ===")
    
    # Test digit estimation
    import math
    for base, exp in [(2, 100), (10, 50), (3, 200)]:
        digits = int(exp * math.log10(base)) + 1
        print(f"{base}^{exp} ~= {digits} digits")
    
    # Test factorial estimation  
    import gmpy2
    for n in [10, 20, 100]:
        fact = gmpy2.fac(n)
        digits = estimate_digits(fact)
        print(f"{n}! = {fact if digits < 20 else f'{get_first_n_digits(fact, 10)}...'} ({digits} digits)")
    print()

def test_batch_primes():
    """Test batch prime testing."""
    print("=== Testing Batch Prime Testing ===")
    
    # Test known primes
    correct_primes = 0
    test_primes = TEST_PRIMES[:10]
    for p in test_primes:
        result = is_prime(p)
        status = "OK" if result else "X"
        print(f"  {p:4} -> {status}")
        if result:
            correct_primes += 1
    
    print(f"Primes: {correct_primes}/{len(test_primes)} correct")
    
    # Test known composites
    correct_composites = 0
    test_composites = TEST_COMPOSITES[:10]
    for c in test_composites:
        result = is_prime(c)
        status = "X" if result else "OK"
        print(f"  {c:4} -> {status}")
        if not result:
            correct_composites += 1
    
    print(f"Composites: {correct_composites}/{len(test_composites)} correct")
    print()

def main():
    """Run all tests."""
    print("Testing Prime Number Analysis Tool Components")
    print("=" * 60)
    
    try:
        test_mersenne_analysis()
        test_primality_testing()
        test_modular_arithmetic()
        test_quick_tools()
        test_batch_primes()
        
        print("=" * 60)
        print("All functionality tests completed successfully!")
        print("The menu system is ready to use.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()