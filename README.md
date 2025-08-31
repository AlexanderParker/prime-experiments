# Prime Number Experiments

Personal playground for exploring prime number algorithms and mathematical patterns.

## Current Algorithm

The main work is in **[test11.ipynb](notebooks/test11.ipynb)** - an algorithm that successfully finds the next prime after any given number using modular arithmetic patterns.

### How It Works

The algorithm uses modular arithmetic to directly calculate the next prime after any given number by finding "blocked slots" where primes cannot exist.

**Core Algorithm (for finding next prime after a known prime):**
1. For a given prime p, find all primes less than √p to use as trial divisors
2. Calculate `-p % prime` for each trial divisor - this gives the distance to the next multiple of that prime
3. These distances represent "blocked slots" where the next prime cannot be located
4. Cycle each trial divisor through multiple iterations to find all blocked slots up to √p
5. Search for the first even number (starting from 2) that isn't in the blocked slots set
6. That unblocked position is the gap to the next prime: `next_prime = p + gap`

**Example with p = 97:**
- Trial divisors: [2, 3, 5, 7] (primes < √97 ≈ 9.8)
- `-97 % 2 = 1` → blocks gaps {1, 3, 5, 7, 9}
- `-97 % 3 = 2` → blocks gaps {2, 5, 8}  
- `-97 % 5 = 3` → blocks gaps {3, 8}
- `-97 % 7 = 1` → blocks gaps {1, 8}
- Combined blocked gaps up to √97: {1, 2, 3, 5, 7, 8, 9}
- First unblocked even gap: 4 → `97 + 4 = 101` ✓

**Generalised Version (for any number n):**
The algorithm can be extended to work from any starting number n, not just known primes. It follows the same process but must ensure it knows all primes up to √n first, then searches from the appropriate starting position (1 if n is even, 2 if n is odd) to maintain the even gap pattern.

**Example with n = 100:**
- Trial divisors: [2, 3, 5, 7] (primes < √100 = 10)
- `-100 % 2 = 0` → blocks gaps {0, 2, 4, 6, 8}
- `-100 % 3 = 2` → blocks gaps {2, 5, 8}  
- `-100 % 5 = 0` → blocks gaps {0, 5}
- `-100 % 7 = 5` → blocks gaps {5}
- Combined blocked gaps up to √100: {0, 2, 4, 5, 6, 8}
- First unblocked gap starting from 1: 1 → `100 + 1 = 101` ✓

**Mathematical Foundation:**
Any composite number must have a prime factor ≤ √n. By using all primes up to √n as trial divisors and cycling their modular patterns, we identify every position where a composite number must occur. The first gap in this pattern is mathematically guaranteed to be prime.

## Proof: Deriving next_prime(p) from p

**Given:** A prime p  
**Goal:** Prove the algorithm correctly finds next_prime(p)

**Theorem:** Let p be prime and let G be the set of all gaps g ≤ √p where g ≡ -p (mod q) for any prime q < √p. Then the smallest even integer g ∉ G satisfies: p + g = next_prime(p).

**Proof:**

**Step 1 - Construction of G:** For each prime q < √p, we calculate r = -p mod q and add all values {r, r+q, r+2q, ...} ≤ √p to G. This gives us G = {all gaps g ≤ √p where g ≡ -p (mod q) for any prime q < √p}.

**Step 2 - Properties of gaps in G:** If g ∈ G, then by construction g ≡ -p (mod q) for some prime q < √p. This modular relationship means that p + g will be divisible by q, so we can assume p + g is composite.

**Step 3 - Completeness of G:** Suppose p + g is composite for some g ≤ √p. Then p + g has a prime factor q. Since p + g ≤ p + √p, if all prime factors of p + g were greater than √p, then p + g could have at most one such factor (otherwise the product would exceed p + g). But a number with exactly one prime factor greater than √p would itself be that prime, contradicting our assumption that p + g is composite. Therefore, p + g must have a prime factor q ≤ √p.

Since q divides p + g, we have p + g ≡ 0 (mod q), which means g ≡ -p (mod q). Since our algorithm constructed G by computing -p mod q for all primes q < √p and cycling through all remainders up to √p, we have g ∈ G.

Therefore, every composite p + g (with g ≤ √p) has its gap g contained in G, proving G's completeness.

**Step 4 - First gap not in G yields a prime:** By Step 3's contrapositive, if g ∉ G (and g ≤ √p), then p + g cannot be composite, so p + g must be prime.

**Step 5 - This prime is next_prime(p):** Since we find the smallest even g ∉ G, and by Steps 2-3 all composite numbers p + g' (for even g' < g, g' ≤ √p) correspond to g' ∈ G, there are no primes between p and p + g. Therefore p + g = next_prime(p).

**Conclusion:** The algorithm correctly identifies next_prime(p) by using modular arithmetic to pre-compute all gaps that must yield composite numbers, then selecting the first gap that avoids these modular constraints. □

## Proof: Generalised Version for Any Number n

**Given:** Any number n  
**Goal:** Prove the generalised algorithm correctly finds next_prime(n)

**Theorem:** Let n be any positive integer and let G be the set of all gaps g ≤ √n where g ≡ -n (mod q) for any prime q < √n. Then the smallest gap g ∉ G (starting from g = 1 if n is even, g = 2 if n is odd) satisfies: n + g = next_prime(n).

**Proof:**

**Step 1 - Construction of G:** For each prime q < √n, we calculate r = -n mod q and add all values {r, r+q, r+2q, ...} ≤ √n to G. This gives us G = {all gaps g ≤ √n where g ≡ -n (mod q) for any prime q < √n}.

**Step 2 - Properties of gaps in G:** If g ∈ G, then by construction g ≡ -n (mod q) for some prime q < √n. This modular relationship means that n + g will be divisible by q, so we can assume n + g is composite.

**Step 3 - Completeness of G:** The same argument as the prime case applies. Any composite n + g (with g ≤ √n) must have a prime factor q ≤ √n, creating the modular relationship g ≡ -n (mod q), which places g in our constructed set G.

**Step 4 - Gap search strategy:** Since gaps between consecutive primes are always even (except the gap from 2 to 3), we search starting from g = 1 if n is even, or g = 2 if n is odd, checking only even values thereafter.

**Step 5 - First valid gap yields next prime:** The first gap g ∉ G that satisfies our search criteria must yield a prime n + g by Step 3's contrapositive, and this must be next_prime(n) since we take the smallest such gap.

**Conclusion:** The generalised algorithm works by the same modular arithmetic principles as the prime-specific version, with the key insight that the blocked gap pattern G captures all composite-yielding positions regardless of whether the starting number is prime. □

## Future Directions

The algorithm opens several interesting research paths:

**Signal Processing Applications:**
- Stacking sin(remainder / n) functions reveals prime patterns
- Could provide entry point for analysing the algorithm with real numbers rather than integers
- Research areas: Fourier analysis of prime distribution, digital signal processing techniques, spectral analysis of modular arithmetic patterns

**Non-Iterative Solutions:**
- Current implementation cycles through remainders iteratively
- Mathematical question: given a set of cycling numbers, can we determine when none equal zero without iteration?
- Research areas: Chinese Remainder Theorem applications, diophantine equations, lattice theory, algebraic number theory

**Additional Mathematical Connections:**
- Sieve theory (relationship to Sieve of Eratosthenes)
- Analytic number theory (connections to prime gap distributions)
- Computational complexity theory (algorithmic efficiency bounds)
- Combinatorial number theory (counting arguments for blocked positions)
- Modular forms and L-functions (deeper arithmetic structure)

## Additional Insights: Prime Blocking Patterns

**Hierarchical Coverage Property:**
The algorithm reveals an interesting structural property about prime distribution. Consider why we only need primes up to √n - taking n = 100, we use primes {2, 3, 5, 7} but not 11.

When 11 creates its blocking pattern, it appears at multiples: 11, 22, 33, 44, 55, 66, 77, 88, 99... This creates 10-position gaps between consecutive multiples. Within each such gap, the blocking patterns from smaller primes {2, 3, 5, 7} provide complete coverage for identifying composite positions.

This suggests a hierarchical structure: each layer of primes creates blocking patterns at its appropriate scale, with smaller primes providing sufficient density to fill the gaps left by larger primes.

**Implications for Twin Primes:**
This blocking structure offers insight into twin prime formation. When a new prime p appears, it blocks its own position but then creates multiples at p×2, p×3, p×4, etc. Critically, these multiples are congruent to existing smaller primes (p×2 ≡ 2, p×3 ≡ 3, etc.), meaning they align with already-blocked positions rather than creating new blocking patterns.

Since new primes never create novel blocking patterns that weren't already covered by smaller primes, they cannot prevent the formation of twin primes in positions that were previously unblocked. This suggests that the introduction of larger primes preserves the spacing patterns established by smaller primes, potentially explaining why twin primes continue to appear throughout the number line.