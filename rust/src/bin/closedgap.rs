// Can the gap be had without iterating over the lattice?
//
// The original algorithm in `rust/src/main.rs` does three things:
//
//   1. for each gear `p <= R = ceil(sqrt(n))`, compute its first tooth as a distance from `n`:
//          first(p) = (p - n mod p) mod p
//      This is already closed form - two modular reductions, nothing walked. It is the core
//      insight and it is correct.
//   2. if any first tooth lands on a candidate offset, mark the whole lattice of teeth
//          first(p), first(p) + p, first(p) + 2p, ...  up to R
//      for every gear.
//   3. scan the marked array for the first unmarked candidate offset.
//
// Steps 2 and 3 are the lattice iteration. Step 2 costs `sum_{p<=R} R/p ~ R log log R` marks and
// step 3 costs up to `R/2` reads, so the whole thing is `O(sqrt(n) log log sqrt(n))` regardless of
// how near the next prime actually is.
//
// The observation this file tests: the *test* for a single offset never needed the lattice either.
// Offset `t` is open exactly when no gear has a tooth there, that is when
//
//      no p <= R divides n + t,
//
// which is `pi(R)` modular reductions with an early exit on the first divisor found - and for a
// composite `n + t` that exit comes almost immediately, since small primes are dense. So instead of
// building the lattice and scanning it, walk the *candidates* and test each one directly. The number
// of candidates is `gap/2`, which near `n = 10^10` averages about 12, against a lattice of 10^5.
//
// Neither version is a closed form for the gap itself. That would need the joint condition across
// all gears at once, whose period is the primorial - the open problem. What is available is a
// closed form for "is this one offset open", and using it per candidate removes the lattice.
//
//     cargo run --release --bin closedgap

use std::time::Instant;

/// Odd primes up to `limit`, by simple sieve.
fn primes_upto(limit: u64) -> Vec<u64> {
    let n = limit as usize + 1;
    let mut sieve = vec![true; n];
    sieve[0] = false;
    if n > 1 {
        sieve[1] = false;
    }
    let mut i = 2usize;
    while i * i < n {
        if sieve[i] {
            let mut j = i * i;
            while j < n {
                sieve[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    (2..n).filter(|&i| sieve[i]).map(|i| i as u64).collect()
}

/// The original shape: mark every tooth of every gear across the window, then scan.
/// Returns `(next prime, marks written, offsets read)`.
fn gap_by_lattice(n: u64, primes: &[u64], r: u64) -> (u64, u64, u64) {
    let mut blocked = vec![false; r as usize + 2];
    let mut marks = 0u64;
    for &p in primes {
        if p > r {
            break;
        }
        let first = (p - n % p) % p;
        let mut t = first;
        while t <= r {
            blocked[t as usize] = true;
            marks += 1;
            t += p;
        }
    }
    let step = 2u64;
    let mut t = if n % 2 == 0 { 1 } else { 2 };
    let mut reads = 0u64;
    while t <= r {
        reads += 1;
        if !blocked[t as usize] {
            return (n + t, marks, reads);
        }
        t += step;
    }
    (0, marks, reads)
}

/// The candidate-first shape: no array, no lattice. For each candidate offset in turn, ask the
/// closed-form question "does any gear have a tooth here", with an early exit on the first hit.
/// Returns `(next prime, divisions performed, candidates tried)`.
fn gap_by_candidates(n: u64, primes: &[u64], r: u64) -> (u64, u64, u64) {
    let step = 2u64;
    let mut t = if n % 2 == 0 { 1 } else { 2 };
    let mut divisions = 0u64;
    let mut candidates = 0u64;
    while t <= r {
        candidates += 1;
        let m = n + t;
        let mut open = true;
        for &p in primes {
            if p > r {
                break;
            }
            divisions += 1;
            if m % p == 0 {
                open = false;
                break;
            }
        }
        if open {
            return (m, divisions, candidates);
        }
        t += step;
    }
    (0, divisions, candidates)
}

fn main() {
    let cases: [u64; 6] = [
        7_213_393_222,
        100_000_000_000,
        1_000_000_000_000,
        10_000_000_000_000,
        100_000_000_000_000,
        1_000_000_000_000_000,
    ];

    let rmax = (*cases.iter().max().unwrap() as f64).sqrt().ceil() as u64 + 2;
    let t0 = Instant::now();
    let primes = primes_upto(rmax);
    println!(
        "sieve to {rmax} gave {} primes in {:.3}s (shared setup, both methods need it)\n",
        primes.len(),
        t0.elapsed().as_secs_f64()
    );

    println!(
        "{:>18} {:>10} {:>6} {:>12} {:>12} {:>10} {:>10} {:>8}",
        "n", "R", "gap", "lattice ops", "cand ops", "lattice s", "cand s", "agree"
    );
    for &n in cases.iter() {
        let r = (n as f64).sqrt().ceil() as u64;

        let t1 = Instant::now();
        let (a, marks, reads) = gap_by_lattice(n, &primes, r);
        let ta = t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        let (b, divisions, cands) = gap_by_candidates(n, &primes, r);
        let tb = t2.elapsed().as_secs_f64();

        println!(
            "{:>18} {:>10} {:>6} {:>12} {:>12} {:>10.6} {:>10.6} {:>8}",
            n,
            r,
            a.saturating_sub(n),
            marks + reads,
            divisions,
            ta,
            tb,
            a == b
        );
        let _ = cands;
    }

    println!();
    for &(base, count) in [
        (1_000_000u64, 50_000u64),
        (10_000_000_000u64, 5_000u64),
        (999_999_000_000u64, 1_000u64),
    ]
    .iter()
    {
        let t = Instant::now();
        let (c, d) = sweep(base, count, &primes);
        println!(
            "sweep from {base} over {c} consecutive n: disagreements = {d}  ({:.2}s)",
            t.elapsed().as_secs_f64()
        );
    }

    println!("\nlattice ops = marks written + offsets read; cand ops = modular reductions performed.");
    println!("Both compute the same next prime. The candidate form never allocates the window.");
}

// Verification sweep: the two methods must agree on every n in a range, not just on the
// six benchmark points.
fn sweep(base: u64, count: u64, primes: &[u64]) -> (u64, u64) {
    let mut checked = 0u64;
    let mut disagreements = 0u64;
    for n in base..base + count {
        let r = (n as f64).sqrt().ceil() as u64;
        let (a, _, _) = gap_by_lattice(n, primes, r);
        let (b, _, _) = gap_by_candidates(n, primes, r);
        checked += 1;
        if a != b {
            disagreements += 1;
            if disagreements <= 3 {
                println!("  DISAGREE at n = {n}: lattice {a}, candidates {b}");
            }
        }
    }
    (checked, disagreements)
}
