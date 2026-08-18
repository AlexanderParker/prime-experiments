// F(2, y) by pruned covering search - ENDPOINT-LAW PRUNED variant of maxgap.rs.
//
// Same problem: F(2, y) = 1 + (longest run of positions coverable by choosing,
// for each odd prime q <= y, one adjacent blocked pair {o, o+1} mod q).
//
// Two sound prunings derived from the endpoint law (Constructor round 9,
// research/topgap_endpoint_law.py), in covering-search form:
//
// 1. MOD-3 ENDPOINT SKIP. At the maximal coverable run M, both bounding
//    positions -1 and M are uncovered (else the run extends), and gear 3 is
//    always used at the max (an unused gear 3 could cover position M). Gear 3
//    misses exactly one residue class mod 3, which must contain both -1 and M,
//    so M = -1 = 2 (mod 3) and F = M + 1 = 0 (mod 3). Since coverability is
//    monotone (a covering of L covers L-1), every L not = 0 (mod 3) below F is
//    coverable and need not be searched. All thirteen known exact values are
//    = 0 mod 3 (33, 48, ..., 309), as is the standing bound territory.
//
// 2. LEFT-TAUT OFFSET EXCLUSION. For EVERY L (not just the max):
//        coverable(L)  <=>  coverable(L) with position -1 left uncovered.
//    (=>: take the maximal run's witness C - it cannot cover -1, else an
//    (M+1)-run is covered, contradicting maximality; C restricted to 0..L-1
//    is a left-taut witness. <=: trivial.) So every gear may exclude the two
//    offsets covering position -1, namely o = q-2 and o = q-1: gear q never
//    covers any position = -1 (mod q). This replaces the mirror-canonical o5
//    halving of the original (reflection maps left-taut to RIGHT-taut
//    coverings, so the two prunings are unsound together): the root branches
//    become o5 in {0, 1, 2} - the same count as the canonical half - and every
//    deeper gear loses its two -1-covering offsets, collapsing the branch
//    factor to <= 1 at every leftmost-uncovered position = -1 or -2 (mod q).
//
// The right-endpoint refinement A(G) mod 35 conditions on the gap length G and
// is only valid at the maximum, not per-L; it is deliberately NOT used (it
// would make the incremental loop unsound).
//
//     cargo run --release --bin maxgap_pruned -- 53 420    // start at L = 420
//     cargo run --release --bin maxgap_pruned -- 53 421 3  // one root branch

const WORDS: usize = 12; // 768 bit masks, enough for runs up to 768

type Mask = [u64; WORDS];

fn set_bit(m: &mut Mask, i: usize) {
    m[i / 64] |= 1u64 << (i % 64);
}

fn get_bit(m: &Mask, i: usize) -> bool {
    m[i / 64] >> (i % 64) & 1 == 1
}

fn union(a: &Mask, b: &Mask) -> Mask {
    let mut r = [0u64; WORDS];
    for i in 0..WORDS {
        r[i] = a[i] | b[i];
    }
    r
}

fn popcount(m: &Mask) -> u32 {
    m.iter().map(|w| w.count_ones()).sum()
}

/// Odd primes up to `limit`, grown by trial division against the list so far.
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

struct Search {
    l: usize,
    witness: std::cell::RefCell<Vec<(usize, usize)>>,
    masks: Vec<Vec<Mask>>, // masks[prime index][offset]
    primes: Vec<usize>,
    dead: std::cell::RefCell<std::collections::HashSet<(Mask, u64)>>,
}

impl Search {
    fn new(l: usize, primes: &[usize]) -> Self {
        let mut masks = Vec::with_capacity(primes.len());
        for &q in primes {
            let mut per_offset = Vec::with_capacity(q);
            for o in 0..q {
                let mut m = [0u64; WORDS];
                for i in 0..l {
                    if i % q == o % q || i % q == (o + 1) % q {
                        set_bit(&mut m, i);
                    }
                }
                per_offset.push(m);
            }
            masks.push(per_offset);
        }
        Search {
            l,
            witness: std::cell::RefCell::new(Vec::new()),
            masks,
            primes: primes.to_vec(),
            dead: std::cell::RefCell::new(std::collections::HashSet::new()),
        }
    }

    /// Can positions 0..l-1 all be blocked left-tautly (position -1 stays
    /// uncovered), using each prime at most once? Equivalent to plain
    /// coverability at every l (see header). Gear 3 pinned at offset 0
    /// (translation symmetry; also forced at the max by left-tautness);
    /// gear 5 pre-assigned to a left-taut offset in {0, 1, 2}.
    fn coverable(&self) -> bool {
        debug_assert_eq!(self.primes[0], 3);
        debug_assert_eq!(self.primes[1], 5);
        let q5 = self.primes[1];
        for o5 in 0..q5 - 2 {
            if self.branch(o5) {
                self.witness.borrow_mut().push((self.primes[1], o5));
                self.witness.borrow_mut().push((self.primes[0], 0));
                return true;
            }
        }
        false
    }

    /// One root branch: the divisor 3 pinned at 0 and the divisor 5 at `o5`.
    fn branch(&self, o5: usize) -> bool {
        let base = self.masks[0][0];
        let start = union(&base, &self.masks[1][o5]);
        self.go(&start, 0b11u64, popcount(&start))
    }

    fn go(&self, covered: &Mask, used: u64, done: u32) -> bool {
        if done as usize == self.l {
            return true;
        }
        if self.dead.borrow().contains(&(*covered, used)) {
            return false;
        }
        // prune: the unused primes must be able to cover what is actually left,
        // where each prime's usable offsets exclude the two covering -1.
        let todo = self.l as u32 - done;
        let mut uncovered = [0u64; WORDS];
        for w in 0..WORDS {
            uncovered[w] = !covered[w];
        }
        for i in self.l..WORDS * 64 {
            uncovered[i / 64] &= !(1u64 << (i % 64));
        }
        let mut capacity = 0u32;
        for i in 0..self.primes.len() {
            if used >> i & 1 == 1 {
                continue;
            }
            let q = self.primes[i];
            let mut top = 0u32;
            for (o, m) in self.masks[i].iter().enumerate() {
                if o >= q - 2 {
                    continue; // left-taut: offsets covering -1 are barred
                }
                let mut c = 0u32;
                for w in 0..WORDS {
                    c += (m[w] & uncovered[w]).count_ones();
                }
                top = top.max(c);
            }
            capacity += top;
            if capacity >= todo {
                break;
            }
        }
        if capacity < todo {
            return false;
        }
        let pos = (0..self.l).find(|&i| !get_bit(covered, i)).unwrap();
        for i in 0..self.primes.len() {
            if used >> i & 1 == 1 {
                continue;
            }
            let q = self.primes[i];
            let offsets = [pos % q, (pos + q - 1) % q];
            for &o in offsets.iter() {
                if o >= q - 2 {
                    continue; // left-taut: gear q never covers positions = -1 mod q
                }
                let next = union(covered, &self.masks[i][o]);
                let gained = popcount(&next);
                if self.go(&next, used | 1 << i, gained) {
                    self.witness.borrow_mut().push((q, o));
                    return true;
                }
            }
        }
        if used.count_ones() <= 6 {
            let mut dead = self.dead.borrow_mut();
            if dead.len() < 4_000_000 {
                dead.insert((*covered, used));
            }
        }
        false
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let y: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(23);
    let mut l: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1);
    let primes = odd_primes_upto(y);
    println!("y = {y}, divisors {:?} [endpoint-law pruned: mod-3 skip + left-taut]", primes);
    if args.get(3).map(|s| s == "witness").unwrap_or(false) {
        // largest coverable run at or above l, with the offsets that achieve it
        // (left-taut search at every L; no mod-3 skip needed here)
        let mut best = l;
        loop {
            let se = Search::new(best + 1, &primes);
            if !se.coverable() {
                break;
            }
            best += 1;
            if best > 700 {
                break;
            }
        }
        let se = Search::new(best, &primes);
        if se.coverable() {
            let mut w = se.witness.borrow().clone();
            w.sort();
            w.dedup();
            println!("largest coverable run {best}, so F(2,{y}) = {}", best + 1);
            println!("offsets used (prime, offset), 3 and 5 pre-assigned:");
            println!("  {:?}", w);
            let unused: Vec<usize> =
                primes.iter().cloned().filter(|q| !w.iter().any(|(p, _)| p == q)).collect();
            println!("  divisors not needed: {:?}", unused);
        }
        return;
    }
    if let Some(arg) = args.get(3) {
        // Single L, single root branch: lets one hard negative be split across runs.
        let o5: usize = arg.parse().unwrap();
        if o5 >= primes[1] - 2 {
            println!("branch o5={o5} at L={l}: skipped, not left-taut");
            return;
        }
        let se = Search::new(l, &primes);
        let cov = se.branch(o5);
        println!("branch o5={o5} at L={l}: {}", if cov { "COVERABLE" } else { "uncoverable" });
        return;
    }
    loop {
        if l % 3 != 0 {
            // F = 0 mod 3 (mod-3 endpoint skip); every non-multiple below F is
            // coverable by monotonicity, no search needed.
            println!("  run of {l} is coverable (mod-3 endpoint law, skipped)");
            l += 1;
            continue;
        }
        let s = Search::new(l, &primes);
        if !s.coverable() {
            println!("F(2,{y}) = {l}   (2F = {}, y^2-y-2 = {})", 2 * l, y * y - y - 2);
            return;
        }
        println!("  run of {l} is coverable");
        l += 1;
        if l > 760 {
            println!("gave up at L = 760");
            return;
        }
    }
}
