// Exhaustive test of the covering-count bound  N(L) <= P (1 - d)^L.
//
// See docs/covering-bound-route.md. Each odd prime q <= y blocks two residues mod q, separated by
// `sep`: 1 in the adjacent frame (the maxgap.rs frame), or 3^{-1} mod q in the t-space frame that
// results from conditioning on gear 3. N(L) counts offset vectors blocking every position of
// [0, L). The bound is what would give F_h(y) <= ceil(log P / -log(1 - d)).
//
// The point of doing this in Rust is scale: gears to 23 means 111,546,435 vectors, and gears to 29
// means 3.2 billion, which Python cannot enumerate. Each vector is summarised by the index of its
// lowest uncovered position, so one pass yields N(L) for every L at once.
//
//     cargo run --release --bin coverbound -- 23 120 adjacent
//     cargo run --release --bin coverbound -- 19 100 tspace

const WORDS: usize = 8; // 512 bit masks, enough for runs up to 512

type Mask = [u64; WORDS];

fn set_bit(m: &mut Mask, i: usize) {
    m[i / 64] |= 1u64 << (i % 64);
}

fn union(a: &Mask, b: &Mask) -> Mask {
    let mut r = [0u64; WORDS];
    for i in 0..WORDS {
        r[i] = a[i] | b[i];
    }
    r
}

/// Index of the lowest zero bit, capped at `lmax`.
fn lowest_zero(m: &Mask, lmax: usize) -> usize {
    for w in 0..WORDS {
        let inv = !m[w];
        if inv != 0 {
            let idx = w * 64 + inv.trailing_zeros() as usize;
            return idx.min(lmax);
        }
    }
    lmax
}

fn odd_primes_upto(limit: usize) -> Vec<usize> {
    let mut primes: Vec<usize> = vec![2];
    let mut n = 3;
    while n <= limit {
        if primes.iter().take_while(|&&p| p * p <= n).all(|&p| n % p != 0) {
            primes.push(n);
        }
        n += 2;
    }
    primes.into_iter().filter(|&p| p > 2).collect()
}

/// Modular inverse of 3 mod q, by trial - q is small.
fn inv3(q: usize) -> usize {
    (1..q).find(|&s| (3 * s) % q == 1).unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let y: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(19);
    let lmax: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(100);
    let tspace = args.get(3).map(|s| s == "tspace").unwrap_or(false);
    assert!(lmax <= WORDS * 64, "lmax exceeds mask width");

    // t-space drops gear 3, since conditioning on it is what produces that frame
    let primes: Vec<usize> = odd_primes_upto(y)
        .into_iter()
        .filter(|&q| !tspace || q != 3)
        .collect();
    let seps: Vec<usize> = primes.iter().map(|&q| if tspace { inv3(q) } else { 1 }).collect();

    let period: u128 = primes.iter().map(|&q| q as u128).product();
    let d: f64 = primes.iter().map(|&q| 1.0 - 2.0 / q as f64).product();
    println!(
        "y = {y}, frame = {}, primes {:?}",
        if tspace { "tspace (sep 3^-1)" } else { "adjacent (sep 1)" },
        primes
    );
    println!("  separations {:?}", seps);
    println!("  P = {period}, d = {d:.9}, vectors to enumerate = {period}");

    // per-prime, per-offset blocked-position masks over [0, lmax)
    let masks: Vec<Vec<Mask>> = primes
        .iter()
        .zip(seps.iter())
        .map(|(&q, &s)| {
            (0..q)
                .map(|o| {
                    let mut m = [0u64; WORDS];
                    for i in 0..lmax {
                        if i % q == o % q || i % q == (o + s) % q {
                            set_bit(&mut m, i);
                        }
                    }
                    m
                })
                .collect()
        })
        .collect();

    // histogram of reach = lowest uncovered position, over all offset vectors
    let mut hist = vec![0u64; lmax + 2];
    let n = primes.len();
    let mut idx = vec![0usize; n];
    let mut partial: Vec<Mask> = vec![[0u64; WORDS]; n + 1];
    let mut level = 0usize;
    loop {
        if level == n {
            hist[lowest_zero(&partial[n], lmax)] += 1;
            // backtrack
            loop {
                if level == 0 {
                    break;
                }
                level -= 1;
                idx[level] += 1;
                if idx[level] < primes[level] {
                    partial[level + 1] = union(&partial[level], &masks[level][idx[level]]);
                    level += 1;
                    break;
                }
                idx[level] = 0;
            }
            if level == 0 {
                break;
            }
            continue;
        }
        partial[level + 1] = union(&partial[level], &masks[level][idx[level]]);
        level += 1;
    }

    // N(L) = number of vectors whose reach is at least L
    let mut suffix = vec![0u64; lmax + 3];
    for l in (0..=lmax + 1).rev() {
        suffix[l] = suffix[l + 1] + hist[l];
    }

    // Step law: N(L)/N(L-1) against 1 - d. The offset-usefulness spread is at most 2 and
    // vanishes relatively like q/L, so the margin should widen as L/q_max grows.
    let qmax = *primes.last().unwrap();
    println!("\n  step law: N(L)/N(L-1) vs 1-d = {:.6}", 1.0 - d);
    println!(
        "  {:>5} {:>8} {:>14} {:>12} {:>10} {:>7}",
        "L", "L/qmax", "N(L)", "step ratio", "margin", "holds"
    );
    let mut step_violations = 0usize;
    for l in 1..=lmax {
        let prev = if l == 1 { period as f64 } else { suffix[l - 1] as f64 };
        if prev == 0.0 {
            break;
        }
        let ratio = suffix[l] as f64 / prev;
        let margin = (1.0 - d) - ratio;
        if margin < -1e-12 {
            step_violations += 1;
        }
        if l <= 3 || l % 10 == 0 {
            println!(
                "  {:>5} {:>8.2} {:>14} {:>12.6} {:>10.6} {:>7}",
                l,
                l as f64 / qmax as f64,
                suffix[l],
                ratio,
                margin,
                margin >= -1e-12
            );
        }
    }
    println!("  step-law violations: {step_violations}");

    println!("\n  {:>5} {:>18} {:>18} {:>10} {:>8}", "L", "N(L)", "P (1-d)^L", "ratio", "holds");
    let mut violations = 0usize;
    let mut worst = 0.0f64;
    let mut first_zero = None;
    for l in 1..=lmax {
        let nl = suffix[l];
        let bound = period as f64 * d.mul_add(0.0, (1.0 - d).powi(l as i32));
        let ratio = if bound > 0.0 { nl as f64 / bound } else { f64::INFINITY };
        if ratio > worst {
            worst = ratio;
        }
        if nl as f64 > bound * (1.0 + 1e-12) {
            violations += 1;
        }
        if nl == 0 && first_zero.is_none() {
            first_zero = Some(l);
        }
        if l <= 12 || nl == 0 && first_zero == Some(l) {
            println!(
                "  {:>5} {:>18} {:>18.3} {:>10.5} {:>8}",
                l,
                nl,
                bound,
                ratio,
                nl as f64 <= bound * (1.0 + 1e-12)
            );
        }
    }
    println!("\n  violations: {violations}, worst ratio: {worst:.6}");
    match first_zero {
        Some(l) => println!("  first L with N(L) = 0: {l}  (so F_h = {l} for this frame)"),
        None => println!("  N(L) never reached 0 within L <= {lmax}"),
    }
}
