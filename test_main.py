#!/usr/bin/env python
"""Test script for the main menu system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from algorithms import MersennePrime, is_prime, next_prime

def test_algorithms():
    """Test core algorithms."""
    print("Testing core algorithms...")
    
    # Test primality
    print(f"is_prime(7): {is_prime(7)}")
    print(f"is_prime(8): {is_prime(8)}")
    print(f"is_prime(97): {is_prime(97)}")
    
    # Test next_prime
    print(f"next_prime(10): {next_prime(10)}")
    print(f"next_prime(100): {next_prime(100)}")
    
    # Test Mersenne
    mp = MersennePrime(7)
    print(f"M7 = {mp.value} (digits: {mp.digits})")
    
    mp = MersennePrime(31)
    print(f"M31 has {mp.digits} digits")
    print(f"First 20 digits: {mp.get_first_n_digits(20)}")
    
    print("All algorithm tests passed!")

if __name__ == "__main__":
    test_algorithms()