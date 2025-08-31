#!/usr/bin/env python
"""
Prime Number Analysis Tool
A comprehensive toolkit for exploring prime numbers with special focus on Mersenne primes.
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from algorithms import MersennePrime, is_prime, next_prime, aks_test, ecpp_test
from utils import estimate_digits, get_first_n_digits, get_last_n_digits
from config.settings import MERSENNE_EXPONENTS, RESULTS_DIR, TEST_PRIMES, TEST_COMPOSITES


class PrimeAnalyzer:
    """Main application for prime number analysis."""
    
    def __init__(self):
        self.running = True
        self.current_mersenne = None
    
    def clear_screen(self):
        """Clear the console screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        """Print the application header."""
        print("=" * 70)
        print("                    PRIME NUMBER ANALYSIS TOOL")
        print("=" * 70)
    
    def get_input(self, prompt: str, input_type=str, allow_empty=False) -> Optional:
        """Get validated user input."""
        while True:
            try:
                user_input = input(prompt).strip()
                if not user_input and allow_empty:
                    return None
                if not user_input:
                    print("Input cannot be empty. Please try again.")
                    continue
                return input_type(user_input)
            except ValueError:
                print(f"Invalid input. Expected {input_type.__name__}. Please try again.")
            except KeyboardInterrupt:
                print("\n\nOperation cancelled.")
                return None
    
    def main_menu(self):
        """Display and handle main menu."""
        while self.running:
            self.clear_screen()
            self.print_header()
            
            print("\n" + "-" * 50)
            print("                    MAIN MENU")
            print("-" * 50)
            print("\n  [1] Mersenne Prime Analysis")
            print("  [2] Primality Testing")
            print("  [3] Modular Arithmetic")
            print("  [4] Quick Tools")
            print("  [5] View Saved Results")
            print("  [6] About")
            print("  [0] Exit")
            print("\n" + "-" * 50)
            
            choice = self.get_input("\nSelect option: ")
            
            if choice == "0":
                self.exit_program()
            elif choice == "1":
                self.mersenne_menu()
            elif choice == "2":
                self.primality_menu()
            elif choice == "3":
                self.modular_menu()
            elif choice == "4":
                self.quick_tools_menu()
            elif choice == "5":
                self.view_results()
            elif choice == "6":
                self.show_about()
            else:
                print("Invalid option. Please try again.")
                input("\nPress Enter to continue...")
    
    def mersenne_menu(self):
        """Mersenne prime analysis menu."""
        while True:
            self.clear_screen()
            print("\n" + "-" * 50)
            print("            MERSENNE PRIME ANALYSIS")
            print("-" * 50)
            print("\n  [1] Analyze Known Mersenne Prime")
            print("  [2] Calculate Custom Mersenne Number")
            print("  [3] Compare Two Mersenne Primes")
            print("  [4] List All Known Mersenne Primes")
            print("  [0] Back to Main Menu")
            print("\n" + "-" * 50)
            
            choice = self.get_input("\nSelect option: ")
            
            if choice == "0":
                break
            elif choice == "1":
                self.analyze_known_mersenne()
            elif choice == "2":
                self.calculate_custom_mersenne()
            elif choice == "3":
                self.compare_mersenne()
            elif choice == "4":
                self.list_mersenne_primes()
            else:
                print("Invalid option.")
                input("\nPress Enter to continue...")
    
    def analyze_known_mersenne(self):
        """Analyze a known Mersenne prime."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("         KNOWN MERSENNE PRIMES")
        print("-" * 50)
        
        # Show selection of known Mersenne primes
        known_list = [
            ("M7", 7, "4th", "127"),
            ("M31", 31, "8th", "2,147,483,647"),
            ("M61", 61, "9th", "2.3 × 10^18"),
            ("M127", 127, "12th", "1.7 × 10^38"),
            ("M521", 521, "13th", "6.9 × 10^156"),
            ("M1279", 1279, "15th", "10^385"),
            ("M19937", 19937, "21st", "10^6,002"),
            ("M44497", 44497, "27th", "10^13,395"),
            ("M86243", 86243, "28th", "10^25,962"),
            ("M82589933", 82589933, "51st", "10^24,862,048")
        ]
        
        print("\nSelect a Mersenne prime to analyze:")
        print("\n  #   Name         Rank    Approximate Value")
        print("  " + "-" * 45)
        for i, (name, exp, rank, value) in enumerate(known_list, 1):
            print(f"  [{i}] {name:12} {rank:6} ~= {value}")
        print(f"\n  [11] Enter custom exponent")
        print(f"  [0]  Cancel")
        
        choice = self.get_input("\nSelect option: ")
        
        if choice == "0":
            return
        elif choice == "11":
            exp = self.get_input("Enter Mersenne exponent: ", int)
            if exp:
                self.analyze_mersenne(exp)
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(known_list):
                    name, exp, rank, _ = known_list[idx]
                    self.analyze_mersenne(exp)
            except (ValueError, IndexError):
                print("Invalid selection.")
        
        input("\nPress Enter to continue...")
    
    def analyze_mersenne(self, exponent: int):
        """Perform detailed Mersenne prime analysis."""
        print(f"\nAnalyzing M{exponent} (2^{exponent} - 1)...")
        print("-" * 40)
        
        try:
            mp = MersennePrime(exponent)
            self.current_mersenne = mp
            
            print(f"Exponent: {exponent:,}")
            print(f"Digits: {mp.digits:,}")
            
            if mp.digits <= 100:
                print(f"Value: {mp.value}")
            else:
                print(f"\nFirst 50 digits:")
                print(f"  {mp.get_first_n_digits(50)}...")
                print(f"\nLast 50 digits:")
                print(f"  ...{mp.get_last_n_digits(50)}")
            
            # Primality check for small exponents
            if exponent < 1000:
                print(f"\nPrimality check: ", end="")
                if is_prime(mp.value):
                    print("PRIME (OK)")
                else:
                    print("COMPOSITE")
            
            # Save option
            save = self.get_input("\nSave analysis to file? (y/n): ")
            if save and save.lower() == 'y':
                filename = RESULTS_DIR / f"M{exponent}_analysis.txt"
                with open(filename, 'w') as f:
                    f.write(f"Mersenne Analysis: M{exponent}\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Formula: 2^{exponent} - 1\n")
                    f.write(f"Exponent: {exponent:,}\n")
                    f.write(f"Digits: {mp.digits:,}\n\n")
                    if mp.digits <= 5000:
                        f.write(f"Value:\n{mp.value}\n")
                    else:
                        f.write(f"First 500 digits:\n{mp.get_first_n_digits(500)}\n\n")
                        f.write(f"Last 500 digits:\n{mp.get_last_n_digits(500)}\n")
                print(f"Saved to: {filename}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    def calculate_custom_mersenne(self):
        """Calculate a custom Mersenne number."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("       CALCULATE CUSTOM MERSENNE NUMBER")
        print("-" * 50)
        print("\nEnter an exponent n to calculate 2^n - 1")
        print("Note: Large exponents may take time to compute")
        
        exp = self.get_input("\nEnter exponent: ", int)
        if exp and exp > 0:
            self.analyze_mersenne(exp)
        
        input("\nPress Enter to continue...")
    
    def compare_mersenne(self):
        """Compare two Mersenne primes."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("         COMPARE MERSENNE PRIMES")
        print("-" * 50)
        
        print("\nEnter two Mersenne exponents to compare")
        exp1 = self.get_input("First exponent: ", int)
        exp2 = self.get_input("Second exponent: ", int)
        
        if exp1 and exp2 and exp1 > 0 and exp2 > 0:
            try:
                print("\nCalculating...")
                mp1 = MersennePrime(exp1)
                mp2 = MersennePrime(exp2)
                
                print("\n" + "=" * 50)
                print(f"{'Property':20} {'M' + str(exp1):^14} {'M' + str(exp2):^14}")
                print("=" * 50)
                print(f"{'Exponent':20} {exp1:^14,} {exp2:^14,}")
                print(f"{'Digits':20} {mp1.digits:^14,} {mp2.digits:^14,}")
                
                if mp1.digits > 0 and mp2.digits > 0:
                    ratio = mp2.digits / mp1.digits if mp1.digits < mp2.digits else mp1.digits / mp2.digits
                    larger = "M" + str(exp2) if mp2.digits > mp1.digits else "M" + str(exp1)
                    print(f"{'Digit Ratio':20} {f'{larger} is {ratio:.2f}x larger':^30}")
                
                if mp1.digits < 100 and mp2.digits < 100:
                    print("\n" + "-" * 50)
                    print(f"M{exp1} = {mp1.value}")
                    print(f"M{exp2} = {mp2.value}")
                    
            except Exception as e:
                print(f"Error: {e}")
        
        input("\nPress Enter to continue...")
    
    def list_mersenne_primes(self):
        """List all known Mersenne primes."""
        self.clear_screen()
        print("\n" + "=" * 70)
        print("                 ALL KNOWN MERSENNE PRIMES")
        print("=" * 70)
        print("\n  #    Name         Exponent        Digits      Discovery")
        print("  " + "-" * 60)
        
        mersenne_data = [
            (1, "M2", 2, 1, "Ancient"),
            (2, "M3", 3, 1, "Ancient"),
            (3, "M5", 5, 2, "Ancient"),
            (4, "M7", 7, 3, "Ancient"),
            (5, "M13", 13, 4, "1456"),
            (6, "M17", 17, 6, "1588"),
            (7, "M19", 19, 6, "1588"),
            (8, "M31", 31, 10, "1772"),
            (9, "M61", 61, 19, "1883"),
            (10, "M89", 89, 27, "1911"),
            (11, "M107", 107, 33, "1914"),
            (12, "M127", 127, 39, "1876"),
            (13, "M521", 521, 157, "1952"),
            (14, "M607", 607, 183, "1952"),
            (15, "M1279", 1279, 386, "1952"),
            (20, "M4423", 4423, 1332, "1961"),
            (25, "M21701", 21701, 6533, "1978"),
            (30, "M132049", 132049, 39751, "1983"),
            (35, "M1398269", 1398269, 420921, "1996"),
            (40, "M13466917", 13466917, 4053946, "2001"),
            (45, "M37156667", 37156667, 11185272, "2008"),
            (50, "M77232917", 77232917, 23249425, "2018"),
            (51, "M82589933", 82589933, 24862048, "2018")
        ]
        
        for num, name, exp, digits, year in mersenne_data:
            if digits < 1000:
                digit_str = str(digits)
            elif digits < 1000000:
                digit_str = f"{digits/1000:.1f}K"
            else:
                digit_str = f"{digits/1000000:.1f}M"
            
            print(f"  {num:2}   {name:12} {exp:12,}   {digit_str:>10}      {year}")
        
        print("\n  Note: Only showing selected entries. Total known: 51")
        
        input("\nPress Enter to continue...")
    
    def primality_menu(self):
        """Primality testing menu."""
        while True:
            self.clear_screen()
            print("\n" + "-" * 50)
            print("            PRIMALITY TESTING")
            print("-" * 50)
            print("\n  [1] Test Single Number")
            print("  [2] Test Number Range")
            print("  [3] Find Next Prime")
            print("  [4] Compare Test Algorithms")
            print("  [0] Back to Main Menu")
            print("\n" + "-" * 50)
            
            choice = self.get_input("\nSelect option: ")
            
            if choice == "0":
                break
            elif choice == "1":
                self.test_single_prime()
            elif choice == "2":
                self.test_range()
            elif choice == "3":
                self.find_next_prime()
            elif choice == "4":
                self.compare_algorithms()
            else:
                print("Invalid option.")
                input("\nPress Enter to continue...")
    
    def test_single_prime(self):
        """Test if a single number is prime."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("         TEST SINGLE NUMBER")
        print("-" * 50)
        
        num = self.get_input("\nEnter number to test: ", int)
        if num and num > 0:
            print(f"\nTesting {num:,}...")
            
            # Test with multiple methods for smaller numbers
            if num < 1000000:
                print("\n" + "─" * 40)
                
                # Fast test
                start = time.time()
                result = is_prime(num)
                fast_time = time.time() - start
                
                print(f"Result: {num:,} is {'PRIME' if result else 'COMPOSITE'}")
                print(f"Test time: {fast_time:.6f} seconds")
                
                # Show factors for small composites
                if not result and num < 10000:
                    factors = []
                    for i in range(2, int(num**0.5) + 1):
                        if num % i == 0:
                            factors.append(i)
                            if len(factors) >= 5:
                                break
                    if factors:
                        print(f"Factors: {factors[:5]}...")
            else:
                result = is_prime(num)
                print(f"\nResult: {num:,} is {'PRIME' if result else 'COMPOSITE'}")
        
        input("\nPress Enter to continue...")
    
    def test_range(self):
        """Test primality for a range of numbers."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("          TEST NUMBER RANGE")
        print("-" * 50)
        
        start = self.get_input("\nStart number: ", int)
        end = self.get_input("End number: ", int)
        
        if start and end and 0 < start <= end:
            # Limit range to prevent long execution
            if end - start > 1000:
                print("\nWarning: Large range. Limiting to first 1000 numbers.")
                end = start + 1000
            
            print(f"\nTesting range {start:,} to {end:,}...")
            print("\n" + "─" * 40)
            
            primes = []
            composites = []
            
            for num in range(start, end + 1):
                if is_prime(num):
                    primes.append(num)
                else:
                    composites.append(num)
            
            print(f"Primes found: {len(primes)}")
            print(f"Composites found: {len(composites)}")
            
            if primes and len(primes) <= 20:
                print(f"\nPrime numbers: {primes}")
            elif primes:
                print(f"\nFirst 10 primes: {primes[:10]}")
                print(f"Last 10 primes: {primes[-10:]}")
            
            # Show prime density
            if end > start:
                density = len(primes) / (end - start + 1) * 100
                print(f"\nPrime density: {density:.2f}%")
        
        input("\nPress Enter to continue...")
    
    def find_next_prime(self):
        """Find the next prime after a given number."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("           FIND NEXT PRIME")
        print("-" * 50)
        
        num = self.get_input("\nEnter starting number: ", int)
        if num and num > 0:
            print(f"\nFinding next prime after {num:,}...")
            
            next_p = next_prime(num)
            gap = next_p - num
            
            print("\n" + "─" * 40)
            print(f"Starting number: {num:,}")
            print(f"Next prime: {next_p:,}")
            print(f"Gap size: {gap}")
            
            # Find a few more for context
            print("\nNext few primes:")
            current = next_p
            for i in range(5):
                current = next_prime(current)
                print(f"  {current:,}")
        
        input("\nPress Enter to continue...")
    
    def compare_algorithms(self):
        """Compare different primality testing algorithms."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("       COMPARE TESTING ALGORITHMS")
        print("-" * 50)
        
        print("\nNote: Advanced algorithms (AKS, ECPP) are limited")
        print("to smaller numbers for performance reasons.")
        
        num = self.get_input("\nEnter number to test (< 10000): ", int)
        if num and 0 < num < 10000:
            print(f"\nTesting {num} with different algorithms...")
            print("\n" + "-" * 50)
            print(f"{'Algorithm':<15} {'Result':<12} {'Time (sec)'}")
            print("-" * 50)
            
            # Test with gmpy2
            start = time.time()
            result1 = is_prime(num)
            time1 = time.time() - start
            print(f"{'GMPY2':<15} {'PRIME' if result1 else 'COMPOSITE':<12} {time1:.6f}")
            
            # Test with AKS
            if num < 1000:
                start = time.time()
                result2 = aks_test(num)
                time2 = time.time() - start
                print(f"{'AKS':<15} {'PRIME' if result2 else 'COMPOSITE':<12} {time2:.6f}")
            
            # Test with ECPP
            if num < 5000:
                start = time.time()
                result3 = ecpp_test(num)
                time3 = time.time() - start
                print(f"{'ECPP':<15} {'PRIME' if result3 else 'COMPOSITE':<12} {time3:.6f}")
        
        input("\nPress Enter to continue...")
    
    def modular_menu(self):
        """Modular arithmetic menu."""
        while True:
            self.clear_screen()
            print("\n" + "-" * 50)
            print("          MODULAR ARITHMETIC")
            print("-" * 50)
            print("\n  [1] Calculate Modular Remainders")
            print("  [2] Find Prime Gaps")
            print("  [3] Analyze Patterns")
            print("  [0] Back to Main Menu")
            print("\n" + "-" * 50)
            
            choice = self.get_input("\nSelect option: ")
            
            if choice == "0":
                break
            elif choice == "1":
                self.calculate_remainders()
            elif choice == "2":
                self.find_gaps()
            elif choice == "3":
                self.analyze_patterns()
            else:
                print("Invalid option.")
                input("\nPress Enter to continue...")
    
    def calculate_remainders(self):
        """Calculate modular remainders."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("      CALCULATE MODULAR REMAINDERS")
        print("-" * 50)
        
        print("\nCalculate remainders for Mersenne prime modulo small primes")
        exp = self.get_input("\nEnter Mersenne exponent: ", int)
        max_prime = self.get_input("Maximum prime to check (default 100): ", int, allow_empty=True)
        if not max_prime:
            max_prime = 100
        
        if exp and exp > 0:
            try:
                print(f"\nCalculating for M{exp}...")
                mp = MersennePrime(exp)
                
                mods = mp.calculate_modular_remainders(max_prime)
                sorted_mods = sorted(mods)
                
                print(f"\nFound {len(mods)} unique even remainders")
                if sorted_mods:
                    print(f"First 20: {sorted_mods[:20]}")
                    
                    # Find missing evens
                    missing = mp.find_missing_evens(sorted_mods)
                    if missing:
                        print(f"\nMissing even numbers: {len(missing)}")
                        print(f"First 10 missing: {missing[:10]}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        input("\nPress Enter to continue...")
    
    def find_gaps(self):
        """Find gaps between consecutive primes."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("         FIND PRIME GAPS")
        print("-" * 50)
        
        start = self.get_input("\nStart from number: ", int)
        count = self.get_input("How many gaps to find (max 50): ", int)
        
        if start and count and start > 0 and count > 0:
            count = min(count, 50)  # Limit to 50
            
            print(f"\nFinding {count} prime gaps starting from {start:,}...")
            print("\n" + "─" * 40)
            print(f"{'Prime':<15} {'Next Prime':<15} {'Gap'}")
            print("-" * 40)
            
            current = start
            gaps = []
            
            for _ in range(count):
                current = next_prime(current)
                next_p = next_prime(current)
                gap = next_p - current
                gaps.append(gap)
                print(f"{current:<15,} {next_p:<15,} {gap:>3}")
                current = next_p
            
            if gaps:
                print("\n" + "─" * 40)
                print(f"Average gap: {sum(gaps)/len(gaps):.2f}")
                print(f"Maximum gap: {max(gaps)}")
                print(f"Minimum gap: {min(gaps)}")
        
        input("\nPress Enter to continue...")
    
    def analyze_patterns(self):
        """Analyze modular patterns."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("        ANALYZE MOD PATTERNS")
        print("-" * 50)
        
        num = self.get_input("\nEnter number to analyze: ", int)
        if num and num > 0:
            print(f"\nModular analysis of {num:,}")
            print("\n" + "─" * 30)
            print(f"{'Modulus':<10} {'Remainder'}")
            print("-" * 30)
            
            for mod in range(2, 21):
                remainder = num % mod
                print(f"{mod:<10} {remainder}")
            
            # Check for interesting patterns
            print("\n" + "─" * 30)
            if num % 2 == 0:
                print("• Even number")
            else:
                print("• Odd number")
            
            if num % 3 == 0:
                print("• Divisible by 3")
            
            if num % 5 == 0:
                print("• Divisible by 5")
        
        input("\nPress Enter to continue...")
    
    def quick_tools_menu(self):
        """Quick tools and utilities menu."""
        while True:
            self.clear_screen()
            print("\n" + "-" * 50)
            print("            QUICK TOOLS")
            print("-" * 50)
            print("\n  [1] Estimate Digits (n^m)")
            print("  [2] Calculate Factorial")
            print("  [3] Generate Random Prime")
            print("  [4] Batch Test Primes")
            print("  [0] Back to Main Menu")
            print("\n" + "-" * 50)
            
            choice = self.get_input("\nSelect option: ")
            
            if choice == "0":
                break
            elif choice == "1":
                self.estimate_digits_tool()
            elif choice == "2":
                self.factorial_tool()
            elif choice == "3":
                self.random_prime_tool()
            elif choice == "4":
                self.batch_test()
            else:
                print("Invalid option.")
                input("\nPress Enter to continue...")
    
    def estimate_digits_tool(self):
        """Estimate digits in large powers."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("         ESTIMATE DIGITS")
        print("-" * 50)
        
        print("\nEstimate digits in n^m")
        base = self.get_input("Enter base (n): ", int)
        exp = self.get_input("Enter exponent (m): ", int)
        
        if base and exp and base > 0 and exp > 0:
            import math
            digits = int(exp * math.log10(base)) + 1
            print(f"\n{base}^{exp} has approximately {digits:,} digits")
            
            # Show some context
            if digits < 100:
                actual = base ** exp
                actual_digits = len(str(actual))
                print(f"Actual: {actual_digits} digits")
                print(f"Value: {actual}")
        
        input("\nPress Enter to continue...")
    
    def factorial_tool(self):
        """Calculate factorial."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("        CALCULATE FACTORIAL")
        print("-" * 50)
        
        n = self.get_input("\nEnter number: ", int)
        
        if n and n >= 0:
            if n > 10000:
                print("Warning: Large factorials may take time...")
            
            import gmpy2
            result = gmpy2.fac(n)
            digits = estimate_digits(result)
            
            print(f"\n{n}! has {digits:,} digits")
            
            if digits <= 100:
                print(f"Value: {result}")
            else:
                print(f"\nFirst 50 digits: {get_first_n_digits(result, 50)}")
                print(f"Last 50 digits: {get_last_n_digits(result, 50)}")
        
        input("\nPress Enter to continue...")
    
    def random_prime_tool(self):
        """Generate a random prime."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("       GENERATE RANDOM PRIME")
        print("-" * 50)
        
        bits = self.get_input("\nNumber of bits (e.g., 128): ", int)
        
        if bits and bits > 0:
            import random
            import gmpy2
            
            print(f"\nGenerating {bits}-bit prime...")
            
            # Generate random odd number and find next prime
            n = random.getrandbits(bits) | 1
            prime = gmpy2.next_prime(n)
            
            print(f"\nGenerated prime:")
            print(f"Value: {prime}")
            print(f"Digits: {estimate_digits(prime)}")
            print(f"Bits: {prime.bit_length()}")
        
        input("\nPress Enter to continue...")
    
    def batch_test(self):
        """Batch test known primes and composites."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("         BATCH PRIME TESTING")
        print("-" * 50)
        
        print("\nTesting known primes...")
        correct_primes = 0
        for p in TEST_PRIMES[:15]:  # Test first 15
            result = is_prime(p)
            status = "OK" if result else "X"
            print(f"  {p:5} -> {status}")
            if result:
                correct_primes += 1
        
        print(f"\nPrimes: {correct_primes}/{len(TEST_PRIMES[:15])} correct")
        
        print("\nTesting known composites...")
        correct_composites = 0
        for c in TEST_COMPOSITES[:15]:  # Test first 15
            result = is_prime(c)
            status = "X" if result else "OK"
            print(f"  {c:5} -> {status}")
            if not result:
                correct_composites += 1
        
        print(f"\nComposites: {correct_composites}/{len(TEST_COMPOSITES[:15])} correct")
        
        input("\nPress Enter to continue...")
    
    def view_results(self):
        """View saved analysis results."""
        self.clear_screen()
        print("\n" + "-" * 50)
        print("         VIEW SAVED RESULTS")
        print("-" * 50)
        
        # List result files
        result_files = list(RESULTS_DIR.glob("*.txt"))
        
        if not result_files:
            print("\nNo saved results found.")
            print("Run some analyses and save them first!")
        else:
            print("\nSaved result files:")
            for i, file in enumerate(result_files[:20], 1):  # Limit to 20 files
                size = file.stat().st_size
                size_str = f"{size/1024:.1f}KB" if size > 1024 else f"{size}B"
                print(f"  [{i}] {file.name:<30} ({size_str})")
            
            if len(result_files) > 20:
                print(f"  ... and {len(result_files)-20} more files")
            
            print(f"\n  [0] Cancel")
            
            choice = self.get_input("\nSelect file to view: ")
            
            if choice and choice != "0":
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(result_files):
                        with open(result_files[idx], 'r') as f:
                            content = f.read(2000)  # Read first 2000 chars
                        
                        print("\n" + "=" * 50)
                        print(content)
                        if len(content) == 2000:
                            print("\n... (file truncated)")
                        print("=" * 50)
                except (ValueError, IndexError):
                    print("Invalid selection.")
        
        input("\nPress Enter to continue...")
    
    def show_about(self):
        """Display information about the application."""
        self.clear_screen()
        print("\n" + "═" * 70)
        print("                        ABOUT")
        print("═" * 70)
        
        print("\n  Prime Number Analysis Tool")
        print("  Version 2.0")
        print("\n  " + "─" * 60)
        
        print("\n  A comprehensive toolkit for exploring prime numbers,")
        print("  with special focus on Mersenne primes (2^p - 1).")
        
        print("\n  FEATURES:")
        print("  • Analysis of Mersenne primes up to M82589933")
        print("  • Multiple primality testing algorithms")
        print("  • Modular arithmetic and pattern analysis")
        print("  • Support for numbers with millions of digits")
        
        print("\n  ALGORITHMS:")
        print("  • GMPY2 (optimized Miller-Rabin)")
        print("  • AKS (deterministic polynomial-time)")
        print("  • ECPP (Elliptic Curve Primality Proving)")
        
        print("\n  MERSENNE PRIMES:")
        print("  • 51 known as of 2018")
        print("  • Largest: M82589933 = 2^82,589,933 - 1")
        print("  • Contains 24,862,048 decimal digits")
        
        print("\n" + "═" * 70)
        
        input("\nPress Enter to continue...")
    
    def exit_program(self):
        """Exit the program."""
        print("\n" + "-" * 50)
        print("Thank you for using Prime Number Analysis Tool!")
        print("Goodbye!")
        print("-" * 50)
        self.running = False
    
    def run(self):
        """Main application entry point."""
        # Show welcome screen
        self.clear_screen()
        print("=" * 70)
        print("           WELCOME TO PRIME NUMBER ANALYSIS TOOL")
        print("=" * 70)
        print("\n  Explore the fascinating world of prime numbers!")
        print("  Special focus on Mersenne primes and advanced algorithms.")
        print("\n  Press Enter to continue...")
        input()
        
        # Run main menu
        self.main_menu()


def main():
    """Program entry point."""
    try:
        app = PrimeAnalyzer()
        app.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please report this issue.")
        sys.exit(1)


if __name__ == "__main__":
    main()