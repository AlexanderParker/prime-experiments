"""Configuration settings for the prime number project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MERSENNE_DIR = DATA_DIR / "mersenne"
RESULTS_DIR = DATA_DIR / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for directory in [DATA_DIR, MERSENNE_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Computation settings
DEFAULT_PRECISION = 1000  # Default precision for gmpy2 computations
MAX_DIGITS_DISPLAY = 100  # Maximum digits to display for large numbers

# Known Mersenne primes (exponents)
MERSENNE_EXPONENTS = {
    "M3": 3,
    "M5": 5,
    "M7": 7,
    "M13": 13,
    "M17": 17,
    "M19": 19,
    "M31": 31,
    "M61": 61,
    "M89": 89,
    "M107": 107,
    "M127": 127,
    "M521": 521,
    "M607": 607,
    "M1279": 1279,
    "M2203": 2203,
    "M2281": 2281,
    "M3217": 3217,
    "M4253": 4253,
    "M4423": 4423,
    "M9689": 9689,
    "M9941": 9941,
    "M11213": 11213,
    "M19937": 19937,
    "M21701": 21701,
    "M23209": 23209,
    "M44497": 44497,
    "M86243": 86243,
    "M110503": 110503,
    "M132049": 132049,
    "M216091": 216091,
    "M756839": 756839,
    "M859433": 859433,
    "M1257787": 1257787,
    "M1398269": 1398269,
    "M2976221": 2976221,
    "M3021377": 3021377,
    "M6972593": 6972593,
    "M13466917": 13466917,
    "M20996011": 20996011,
    "M24036583": 24036583,
    "M25964951": 25964951,
    "M30402457": 30402457,
    "M32582657": 32582657,
    "M37156667": 37156667,
    "M42643801": 42643801,
    "M43112609": 43112609,
    "M57885161": 57885161,
    "M74207281": 74207281,
    "M77232917": 77232917,
    "M82589933": 82589933,  # 51st known Mersenne prime (as of 2018)
}

# Test configuration
TEST_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
TEST_COMPOSITES = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25]

# Performance settings
BATCH_SIZE = 1000  # Number of primes to check in batch operations
PROGRESS_INTERVAL = 100  # Show progress every N iterations