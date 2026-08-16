// F(2, y) by pruned covering search.
//
// A gap of size G in the twin survivor pattern is a run of G - 1 consecutive
// blocked positions. Each odd prime q <= y blocks one pair of adjacent residues
// {o, o + 1} mod q, with the offset o chosen once. So
//
//     F(2, y) = 1 + (longest run of positions coverable that way)
//
// which is a search over offsets rather than a walk over the primorial period.
// The search is exhaustive at the leftmost uncovered position, so the answer is
// exact: if no assignment covers a run of length L, none exists.
//
//     cargo run --release --bin maxgap -- 29        // search from L = 1
//     cargo run --release --bin maxgap -- 29 120    // start at L = 120
//
// Usage note: the printed F is exact. The search reports the first L that cannot
// be covered, and every smaller L was verified coverable on the way up.

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
    /// offsets of the covering found, when one exists (prime index -> offset)
    witness: std::cell::RefCell<Vec<(usize, usize)>>,
    masks: Vec<Vec<Mask>>, // masks[prime index][offset]
    best: Vec<u32>,        // best coverage of each prime
    primes: Vec<usize>,
    // states already proved hopeless; different assignment orders reach the same
    // (covered, used) state, so without this the search redoes the same work
    dead: std::cell::RefCell<std::collections::HashSet<(Mask, u64)>>,
}

impl Search {
    fn new(l: usize, primes: &[usize]) -> Self {
        let mut masks = Vec::with_capacity(primes.len());
        let mut best = Vec::with_capacity(primes.len());
        for &q in primes {
            let mut per_offset = Vec::with_capacity(q);
            let mut top = 0;
            for o in 0..q {
                let mut m = [0u64; WORDS];
                for i in 0..l {
                    if i % q == o % q || i % q == (o + 1) % q {
                        set_bit(&mut m, i);
                    }
                }
                top = top.max(popcount(&m));
                per_offset.push(m);
            }
            masks.push(per_offset);
            best.push(top);
        }
        Search {
            l,
            witness: std::cell::RefCell::new(Vec::new()),
            masks,
            best,
            primes: primes.to_vec(),
            dead: std::cell::RefCell::new(std::collections::HashSet::new()),
        }
    }

    /// Can positions 0..l-1 all be blocked, using each prime at most once?
    fn coverable(&self) -> bool {
        // Translation symmetry: shifting every offset by the same t maps coverings to
        // coverings of a translated run, and the largest gap is translation invariant
        // over the period. Choosing t so that the divisor 3 lands on offset 0 is
        // therefore free, and it cuts the search by a factor of 3 while covering
        // positions 0 and 1 before the search starts.
        debug_assert_eq!(self.primes[0], 3);
        debug_assert_eq!(self.primes[1], 5);
        // Reflection symmetry: reversing the run maps q's blocked pair {o, o+1} to
        // {L-2-o, L-1-o}, another adjacent pair, so coverability is reversal
        // invariant. With the divisor 3 pinned at offset 0, the residual symmetry is
        // the involution R(o) = (L - 2 - o) - s (mod q) with s = (L - 2) mod 3. It is
        // broken by pre-assigning the divisor 5 and keeping only the canonical half of
        // its offsets. Pre-assigning is sound because every configuration has some
        // offset for 5, whether or not the search would have used it.
        let q5 = self.primes[1];
        for o5 in 0..q5 {
            if !self.canonical_o5(o5) {
                continue;
            }
            if self.branch(o5) {
                // record the pre-assigned divisors too, or the witness misreports them
                self.witness.borrow_mut().push((self.primes[1], o5));
                self.witness.borrow_mut().push((self.primes[0], 0));
                return true;
            }
        }
        false
    }

    /// Is this offset for the divisor 5 the canonical one of its mirror pair?
    fn canonical_o5(&self, o5: usize) -> bool {
        let q5 = self.primes[1];
        let s = (self.l + 1) % 3; // (L - 2) mod 3, without underflow
        let mirrored = ((self.l + 2 * q5 - 2 - o5) - s % q5) % q5;
        o5 <= mirrored
    }

    /// One root branch: the divisor 3 pinned at 0 and the divisor 5 at `o5`.
    fn branch(&self, o5: usize) -> bool {
        let base = self.masks[0][0];
        let start = union(&base, &self.masks[1][o5]);
        self.go(&start, 0b11u64, popcount(&start))
    }

    #[allow(dead_code)]
    fn unused(&self) -> bool {
        let base = self.masks[0][0];
        let q5 = self.primes[1];
        let s = (self.l + 1) % 3;
        for o5 in 0..q5 {
            let mirrored = ((self.l + 2 * q5 - 2 - o5) - s % q5) % q5;
            if o5 > mirrored {
                continue;
            }
            let start = union(&base, &self.masks[1][o5]);
            if self.go(&start, 0b11u64, popcount(&start)) {
                return true;
            }
        }
        false
    }

    fn go(&self, covered: &Mask, used: u64, done: u32) -> bool {
        if done as usize == self.l {
            return true;
        }
        if self.dead.borrow().contains(&(*covered, used)) {
            return false;
        }
        // prune: the unused primes must be able to cover what is actually left.
        // Bounding each prime by its best coverage of the *current* uncovered set is
        // far tighter than bounding it by its best coverage of the whole run.
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
            let mut top = 0u32;
            for m in &self.masks[i] {
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
                let next = union(covered, &self.masks[i][o]);
                let gained = popcount(&next);
                if self.go(&next, used | 1 << i, gained) {
                    self.witness.borrow_mut().push((q, o));
                    return true;
                }
            }
        }
        // Bounded memo: only states near the root are worth remembering, and the
        // set is capped so a deep search cannot exhaust memory (an unbounded memo
        // reached 7 GB at y = 37).
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
    println!("y = {y}, divisors {:?}", primes);
    if args.get(3).map(|s| s == "witness").unwrap_or(false) {
        // largest coverable run at or above l, with the offsets that achieve it
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
            println!("offsets used (prime, offset), 5 pre-assigned by symmetry:");
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
        let se = Search::new(l, &primes);
        if !se.canonical_o5(o5) {
            println!("branch o5={o5} at L={l}: skipped, not canonical");
            return;
        }
        let cov = se.branch(o5);
        println!("branch o5={o5} at L={l}: {}", if cov { "COVERABLE" } else { "uncoverable" });
        return;
    }
    loop {
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
