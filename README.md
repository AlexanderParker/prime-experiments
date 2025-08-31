# Prime Number Analysis Project

A comprehensive Python project for analyzing prime numbers, with a special focus on Mersenne primes and various primality testing algorithms.

## Project Structure

```
primes/
├── src/
│   ├── algorithms/       # Primality testing algorithms
│   │   ├── __init__.py
│   │   ├── primality.py  # Basic primality tests
│   │   ├── mersenne.py   # Mersenne prime operations
│   │   ├── aks.py        # AKS primality test
│   │   └── ecpp.py       # Elliptic Curve Primality Proving
│   └── utils/            # Utility functions
│       ├── __init__.py
│       ├── digits.py     # Large number digit operations
│       └── primes.py     # Prime counting functions
├── data/
│   ├── mersenne/         # Mersenne prime data files
│   └── results/          # Analysis results
├── notebooks/            # Jupyter notebooks for experimentation
├── tests/               # Unit tests
├── config/
│   └── settings.py      # Configuration settings
├── main.py              # Main CLI application
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Features

- **Mersenne Prime Analysis**: Calculate and analyze Mersenne primes of any size
- **Multiple Primality Tests**: 
  - Fast gmpy2-based testing
  - AKS deterministic polynomial-time test
  - Elliptic Curve Primality Proving (ECPP)
- **Modular Arithmetic**: Calculate modular remainders and find prime gaps
- **Large Number Handling**: Efficient operations on numbers with millions of digits
- **CLI Interface**: Easy-to-use command-line interface for all operations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd primes
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: Installing `gmpy2` on Windows may require additional steps. Consider using conda:
```bash
conda install gmpy2
```

## Usage

### Quick Start

The tool launches an interactive menu by default:

```bash
# Launch interactive menu (DEFAULT)
python main.py

# Or use the direct menu script
python menu.py

# Or use the launcher scripts
./run_menu.bat  # Windows
./run_menu.sh   # Unix/Linux/Mac
```

### Interactive Menu Features

The interactive menu is the primary interface and provides:

- **Mersenne Prime Analysis**
  - Analyze known Mersenne primes (all 51 known primes)
  - Calculate custom Mersenne numbers
  - Compare different Mersenne primes
  - View first/last digits of huge primes

- **Primality Testing**
  - Test single numbers or ranges
  - Compare algorithms (gmpy2, AKS, ECPP)
  - Find next prime after any number
  - Benchmark performance

- **Modular Arithmetic**
  - Calculate modular remainders
  - Find prime gaps
  - Analyze modular patterns

- **Batch Operations**
  - Test known primes/composites
  - Run performance benchmarks

- **Utilities**
  - Estimate digits in large numbers
  - Extract first/last digits
  - Calculate factorials
  - Generate random primes

- **Results Viewer**
  - Browse saved analysis results
  - Export findings

### Command Line Interface (CLI Mode)

For scripting and automation, you can bypass the menu using direct commands:

#### Analyze a Mersenne Prime
```bash
# Analyze M127 (2^127 - 1)
python main.py mersenne 127

# Analyze and save results
python main.py mersenne 8191 --save
```

#### Calculate Modular Remainders
```bash
# Find modular remainders for M127
python main.py modular 127

# Check more primes (up to 5000)
python main.py modular 127 --max 5000
```

#### Test Primality
```bash
# Test if a number is prime (using gmpy2)
python main.py test 997

# Use different methods
python main.py test 997 --method aks
python main.py test 997 --method ecpp
```

#### Batch Testing
```bash
# Run batch primality tests
python main.py batch
```

#### List Known Mersenne Primes
```bash
# Show all known Mersenne prime exponents
python main.py list
```

### Python API

You can also use the modules directly in Python:

```python
from src.algorithms import MersennePrime, is_prime
from src.utils import estimate_digits

# Create a Mersenne prime
mp = MersennePrime(127)
print(f"M127 has {mp.digits} digits")
print(f"First 50 digits: {mp.get_first_n_digits(50)}")
print(f"Last 50 digits: {mp.get_last_n_digits(50)}")

# Test primality
print(is_prime(997))  # True
print(is_prime(1000)) # False

# Calculate modular remainders
mods = mp.calculate_modular_remainders(1000)
print(f"Found {len(mods)} unique modular remainders")
```

## Algorithms

### Mersenne Primes
Mersenne primes are prime numbers of the form 2^p - 1, where p is also prime. This project includes tools for:
- Calculating Mersenne numbers
- Analyzing their properties
- Finding modular patterns

### Primality Tests

1. **Trial Division**: Basic method for small numbers
2. **GMPY2**: Optimized Miller-Rabin and other tests
3. **AKS Test**: Deterministic polynomial-time algorithm
4. **ECPP**: Elliptic Curve Primality Proving for large numbers

## Data Files

The `data/` directory contains:
- **mersenne/**: Large Mersenne prime numbers (including M82589933 with 24.8 million digits)
- **results/**: Generated analysis results, modular remainders, and prime gaps

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/ main.py
flake8 src/ tests/ main.py
```

### Jupyter Notebooks
Experimental code and visualizations are in the `notebooks/` directory:
```bash
jupyter notebook
```

## Performance Considerations

- The project uses `gmpy2` for efficient arbitrary-precision arithmetic
- Large Mersenne primes (like M82589933) require significant memory
- Modular remainder calculations are optimized for batch processing
- Progress indicators show status for long-running operations

## Known Mersenne Primes

As of 2018, there are 51 known Mersenne primes. The largest is M82589933 (2^82589933 - 1) with 24,862,048 digits. This project includes tools to work with all known Mersenne primes.

## Contributing

Contributions are welcome! Areas for improvement:
- Additional primality testing algorithms
- Performance optimizations
- Visualization tools
- Extended documentation

## License

This project is for educational and research purposes.

## References

- [GIMPS - Great Internet Mersenne Prime Search](https://www.mersenne.org/)
- [The AKS Primality Test](https://en.wikipedia.org/wiki/AKS_primality_test)
- [Elliptic Curve Primality Proving](https://en.wikipedia.org/wiki/Elliptic_curve_primality)
- [GMPY2 Documentation](https://gmpy2.readthedocs.io/)