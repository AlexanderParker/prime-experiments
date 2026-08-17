// F2(2, y): the largest sum of two adjacent gaps, by pruned covering search.
//
// Two adjacent gaps `a` and `b` share an exposed position `p`. The blocked positions are the `a-1`
// before `p` and the `b-1` after it, so the window runs from `p-(a-1)` to `p+(b-1)`, has length
// `L = a+b-1`, and has a hole at index `a-1` that must be left uncovered by every divisor. Hence
//
//     F2(2, y) = 1 + (longest L such that some interior hole h admits a covering of [0,L) \ {h}
//                     with no divisor covering h)
//
// which is the same kind of search as maxgap.rs with two changes: one position must stay uncovered,
// and each divisor loses the 2 offsets that would cover it.
//
// Why this quantity: F(M + q) >= F2(M) always, since every exposed point of M is deleted in exactly 2
// of the q laps when gear q is added (docs/gear-recursion.md section 3). So F2 - F is a lower bound on
// every increment, and whether it stays small decides whether the maximum gap can grow polynomially.
// Measured from the pattern directly it reads 2, 4, 5, 7, 6, 5, 12 in k-units for y = 7..29; past
// y = 29 the pattern no longer fits in memory and this search takes over.
//
//     cargo run --release --bin holegap -- 23          // search from L = 1
//     cargo run --release --bin holegap -- 29 100      // start at L = 100
//
// The printed F2 is exact: the search is exhaustive at the leftmost uncovered position, so if no
// (hole, offsets) assignment covers a window of length L then none exists.
//
// Note the divisor 3 has no choice here. It must avoid the hole, so its offset is forced to
// o = h+1 mod 3, and the whole residue class h mod 3 is left to the divisors >= 5 - except h itself,
// which is the single position of that class needing no cover. That is the mechanical reason the hole
// buys little: one position of slack, against losing the free choice of class and 2 offsets per gear.

const WORDS: usize = 12; // 768 bit masks, enough for windows up to 768

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

fn intersect(a: &Mask, b: &Mask) -> Mask {
    let mut r = [0u64; WORDS];
    for i in 0..WORDS {
        r[i] = a[i] & b[i];
    }
    r
}

fn popcount(m: &Mask) -> u32 {
    m.iter().map(|w| w.count_ones()).sum()
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

struct Search {
    l: usize,
    target: Mask,   // the positions that must be covered: [0,l) minus the hole
    need: u32,      // popcount of target
    masks: Vec<Vec<Mask>>, // masks[prime index][offset], empty mask for disallowed offsets
    allowed: Vec<Vec<usize>>, // offsets that do not cover the hole
    primes: Vec<usize>,
    dead: std::cell::RefCell<std::collections::HashSet<(Mask, u64)>>,
}

impl Search {
    fn new(l: usize, hole: usize, primes: &[usize]) -> Self {
        let mut target = [0u64; WORDS];
        for i in 0..l {
            if i != hole {
                set_bit(&mut target, i);
            }
        }
        let mut masks = Vec::with_capacity(primes.len());
        let mut allowed = Vec::with_capacity(primes.len());
        for &q in primes {
            let mut per_offset = Vec::with_capacity(q);
            let mut ok = Vec::new();
            for o in 0..q {
                // this offset covers the hole when hole = o or o+1 mod q
                let covers_hole = hole % q == o % q || hole % q == (o + 1) % q;
                let mut m = [0u64; WORDS];
                if !covers_hole {
                    for i in 0..l {
                        if i % q == o % q || i % q == (o + 1) % q {
                            set_bit(&mut m, i);
                        }
                    }
                    ok.push(o);
                }
                per_offset.push(m);
            }
            masks.push(per_offset);
            allowed.push(ok);
        }
        let need = popcount(&target);
        Search {
            l,
            target,
            need,
            masks,
            allowed,
            primes: primes.to_vec(),
            dead: std::cell::RefCell::new(std::collections::HashSet::new()),
        }
    }

    /// Can [0,l) \ {hole} be covered with the hole left uncovered?
    fn coverable(&self) -> bool {
        // The divisor 3 has exactly one admissible offset, o = hole+1 mod 3, so it is forced rather
        // than searched. Assert that, since it is the load-bearing simplification.
        debug_assert_eq!(self.primes[0], 3);
        debug_assert_eq!(self.allowed[0].len(), 1);
        let o3 = self.allowed[0][0];
        let start = self.masks[0][o3];
        let done = popcount(&intersect(&start, &self.target));
        self.go(&start, 1u64, done)
    }

    fn go(&self, covered: &Mask, used: u64, done: u32) -> bool {
        if done == self.need {
            return true;
        }
        if self.dead.borrow().contains(&(*covered, used)) {
            return false;
        }
        let todo = self.need - done;
        // what is still needed: target minus covered
        let mut uncovered = [0u64; WORDS];
        for w in 0..WORDS {
            uncovered[w] = self.target[w] & !covered[w];
        }
        // prune: bound each unused divisor by its best coverage of what is actually left
        let mut capacity = 0u32;
        for i in 0..self.primes.len() {
            if used >> i & 1 == 1 {
                continue;
            }
            let mut top = 0u32;
            for &o in &self.allowed[i] {
                let m = &self.masks[i][o];
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
        // branch on the leftmost position still needing cover
        let pos = (0..self.l).find(|&i| get_bit(&self.target, i) && !get_bit(covered, i)).unwrap();
        for i in 0..self.primes.len() {
            if used >> i & 1 == 1 {
                continue;
            }
            let q = self.primes[i];
            for &o in [pos % q, (pos + q - 1) % q].iter() {
                if !self.allowed[i].contains(&o) {
                    continue; // this offset would cover the hole
                }
                let next = union(covered, &self.masks[i][o]);
                let gained = popcount(&intersect(&next, &self.target));
                if self.go(&next, used | 1 << i, gained) {
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

/// Is a window of length `l` coverable for some interior hole?
fn any_hole(l: usize, primes: &[usize]) -> Option<usize> {
    // Reflection: (l, h) and (l, l-1-h) are mirror images, so only half the holes need testing.
    for hole in 1..=(l - 1) / 2 {
        if hole + 1 > l {
            break;
        }
        let se = Search::new(l, hole, primes);
        if se.coverable() {
            return Some(hole);
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let y: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(23);
    let mut l: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(3);
    let primes = odd_primes_upto(y);
    println!("y = {y}, divisors {:?}", primes);
    assert!(l >= 3, "a window needs an interior hole");
    loop {
        assert!(l < WORDS * 64, "window exceeds mask width");
        match any_hole(l, &primes) {
            Some(h) => {
                println!("  window of {l} coverable with hole at {h}");
                l += 1;
            }
            None => {
                println!("F2(2,{y}) = {}   (no window of {l} works for any interior hole)", l);
                return;
            }
        }
    }
}
