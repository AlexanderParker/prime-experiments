// jkcov6.rs -- REDUCED-LATTICE exact search for the k-class Jacobsthal family.
//
// HARVESTER lane, round 28.  Second, independent engine (jkcover.rs is the
// unreduced reference).
//
// REDUCTION.  In the covering restatement
//     j_k(P(z)) - 1 = longest interval coverable by |S_p| <= min(k, p-1),
// every prime p <= k+1 has cap p-1, i.e. it kills all but one class, so the
// survivors lie in a single class mod p and the whole problem rescales.  With
//     D = prod of primes p <= k+1        (D = 2 at k = 1, D = 6 at k = 2, 3)
// we get
//     j_k(P(z)) = D * (m + 1),
//     m = longest run [1, m] coverable by k NON-ZERO classes mod p for each
//         prime k+1 < p <= z.
// k = 1 is Hagedorn's h(n+1) = 2 w(n) + 2; k = 2 is Ziller-Morack's
// h_2 = 6 omega_2 + 6.  Class 0 is excluded because a MAXIMAL covered run has
// an uncovered position on each side, and translating that uncovered position
// to 0 forbids the class 0 at every prime.
//
// SEARCH.  Branch on "which prime covers the leftmost uncovered position".
// CANONICAL FORM (Ziller-Morack Prop. 2.1 / their RPA2 rule "select the
// smallest prime if there are any at choice"): committing prime p at position
// j is REJECTED when an earlier commit (j', p') has j' == j (mod p) and
// p' > p -- p was free then and its class covers j', so the same class set is
// reachable by committing p first.  This removes the (2n-4)!/2^(n-2)
// permutation redundancy of the naive tree.
//
// PRUNING.  U = positions still uncovered in [j, target).  Necessary:
//   (v0) every x in U admits some free prime p with p does not divide x;
//   (v1) sum_p f_p * ceil(M/p) >= |U|;
//   (v2) sum_p (the f_p largest |U cap (r mod p)|, r != 0) >= |U|.
//
// Usage: jkcov6 <k> <z> [--lmax N] [--seed M] [--secs S] [--quiet]
//               [--split S --part I --nparts N]

use std::env;
use std::time::Instant;

const MW: usize = 24; // 1536 positions
type Bits = [u64; MW];

fn primes_upto(n: usize) -> Vec<usize> {
    if n < 2 {
        return vec![];
    }
    let mut s = vec![true; n + 1];
    s[0] = false;
    s[1] = false;
    let mut i = 2;
    while i * i <= n {
        if s[i] {
            let mut j = i * i;
            while j <= n {
                s[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    (2..=n).filter(|&i| s[i]).collect()
}

struct Solver {
    ps: Vec<usize>,
    cap: usize,
    n: usize,
    m: usize, // window: positions 1..=m
    nw: usize,
    masks: Vec<Vec<Bits>>, // masks[i][r], r != 0 only meaningful
    cnt: Vec<usize>,
    cj: Vec<usize>, // commit positions
    cp: Vec<usize>, // commit prime indices
    ncom: usize,
    best: usize,
    bestsol: Vec<Vec<usize>>,
    nodes: u64,
    pv0: u64,
    pv1: u64,
    pv2: u64,
    psym: u64,
    t0: Instant,
    limit: f64,
    aborted: bool,
    unc: Vec<usize>,
    scratch: Vec<(i64, usize)>,
    hist: Vec<u32>,
    hoff: Vec<usize>,
    freei: Vec<usize>,
    freef: Vec<usize>,
    nfree: usize,
    topv: Vec<[u32; 8]>,
    topr: Vec<[usize; 8]>,
    // subtree splitting
    split_depth: usize,
    part: usize,
    nparts: usize,
    leafctr: u64,
}

impl Solver {
    fn new(k: usize, z: usize, m: usize) -> Solver {
        let all = primes_upto(z);
        let ps: Vec<usize> = all.into_iter().filter(|&p| p > k + 1).collect();
        let n = ps.len();
        let nw = (m + 64) / 64;
        assert!(nw <= MW, "window too large");
        assert!(k <= 8, "top-f trackers are sized for k <= 8");
        let mut masks = Vec::with_capacity(n);
        for &p in &ps {
            let mut mm = Vec::with_capacity(p);
            for r in 0..p {
                let mut b: Bits = [0u64; MW];
                if r != 0 {
                    let mut j = r;
                    while j <= m {
                        b[j >> 6] |= 1u64 << (j & 63);
                        j += p;
                    }
                }
                mm.push(b);
            }
            masks.push(mm);
        }
        let mut hoff = Vec::with_capacity(n);
        let mut tot = 0usize;
        for &p in &ps {
            hoff.push(tot);
            tot += p;
        }
        Solver {
            hoff,
            hist: vec![0u32; tot + 1],
            freei: vec![0; n],
            freef: vec![0; n],
            nfree: 0,
            topv: vec![[0u32; 8]; n],
            topr: vec![[usize::MAX; 8]; n],
            ps,
            cap: k,
            n,
            m,
            nw,
            masks,
            cnt: vec![0; n],
            cj: vec![0; 4 * n + 8],
            cp: vec![0; 4 * n + 8],
            ncom: 0,
            best: 0,
            bestsol: vec![Vec::new(); n],
            nodes: 0,
            pv0: 0,
            pv1: 0,
            pv2: 0,
            psym: 0,
            t0: Instant::now(),
            limit: f64::INFINITY,
            aborted: false,
            unc: Vec::with_capacity(m + 1),
            scratch: Vec::new(),
            split_depth: 0,
            part: 0,
            nparts: 1,
            leafctr: 0,
        }
    }

    #[inline]
    fn first_unset(&self, cov: &Bits) -> usize {
        for w in 0..self.nw {
            let mut v = !cov[w];
            if w == 0 {
                v &= !1u64; // position 0 is never a target
            }
            if v != 0 {
                let j = (w << 6) + v.trailing_zeros() as usize;
                if j <= self.m {
                    return j;
                }
                return self.m + 1;
            }
        }
        self.m + 1
    }

    // v3: the PREFIX-WINDOW capacity criterion.  For EVERY prefix [j, x] of the
    // residual window, the number of uncovered positions must not exceed the
    // capacity of the free classes RESTRICTED TO THAT PREFIX.  Short prefixes
    // are where large primes are weakest (one position per class), so this is
    // the sliding form of Hagedorn's m_i bound and subsumes the two global
    // counting bounds.  Cost O(span * #freeprimes), one pass.
    fn feasible_to(&mut self, cov: &Bits, j: usize, target: usize) -> bool {
        let nf = self.nfree;
        if nf == 0 {
            // nothing left to commit: every position in the window must be covered
            let mut x = j;
            while x <= target {
                if (cov[x >> 6] >> (x & 63)) & 1 == 0 {
                    self.pv1 += 1;
                    return false;
                }
                x += 1;
            }
            return true;
        }
        // reset scratch for the free primes only
        for t in 0..nf {
            let i = self.freei[t];
            let p = self.ps[i];
            let base = self.hoff[i];
            for r in 0..p {
                self.hist[base + r] = 0;
            }
            self.topv[t] = [0u32; 8];
            self.topr[t] = [usize::MAX; 8];
        }
        // incremental residues: no division in the inner loop
        let mut pp: [usize; 96] = [1; 96];
        let mut rr: [usize; 96] = [0; 96];
        let mut bb: [usize; 96] = [0; 96];
        for t in 0..nf {
            let i = self.freei[t];
            pp[t] = self.ps[i];
            rr[t] = j % self.ps[i];
            bb[t] = self.hoff[i];
        }
        let mut u: usize = 0;
        let mut capsum: usize = 0;
        let mut x = j;
        while x <= target {
            if (cov[x >> 6] >> (x & 63)) & 1 != 0 {
                for t in 0..nf {
                    rr[t] += 1;
                    if rr[t] == pp[t] {
                        rr[t] = 0;
                    }
                }
                x += 1;
                continue;
            }
            u += 1;
            let mut any = false;
            for t in 0..nf {
                let p = pp[t];
                let r = rr[t];
                rr[t] = if r + 1 == p { 0 } else { r + 1 };
                if r == 0 {
                    continue;
                }
                any = true;
                let base = bb[t];
                self.hist[base + r] += 1;
                let v = self.hist[base + r];
                let f = self.freef[t];
                // maintain the f largest counts over DISTINCT residues
                let tv = &mut self.topv[t];
                let tr = &mut self.topr[t];
                let mut pos = usize::MAX;
                for q in 0..f {
                    if tr[q] == r {
                        pos = q;
                        break;
                    }
                }
                if pos != usize::MAX {
                    capsum += 1;
                    tv[pos] = v;
                    // bubble up
                    let mut q = pos;
                    while q > 0 && tv[q - 1] < tv[q] {
                        tv.swap(q - 1, q);
                        tr.swap(q - 1, q);
                        q -= 1;
                    }
                } else if v > tv[f - 1] {
                    capsum += (v - tv[f - 1]) as usize;
                    tv[f - 1] = v;
                    tr[f - 1] = r;
                    let mut q = f - 1;
                    while q > 0 && tv[q - 1] < tv[q] {
                        tv.swap(q - 1, q);
                        tr.swap(q - 1, q);
                        q -= 1;
                    }
                }
            }
            if !any {
                self.pv0 += 1;
                return false;
            }
            if u > capsum {
                self.pv2 += 1;
                return false;
            }
            x += 1;
        }
        true
    }

    fn dfs(&mut self, cov: Bits, depth: usize) {
        self.nodes += 1;
        if self.nodes & 0x3FFFF == 0 && self.t0.elapsed().as_secs_f64() > self.limit {
            self.aborted = true;
        }
        if self.aborted {
            return;
        }
        let j = self.first_unset(&cov);
        if j > self.m {
            self.best = self.m;
            self.bestsol = self.solution();
            return;
        }
        if j - 1 > self.best {
            self.best = j - 1;
            self.bestsol = self.solution();
        }
        let target = self.best + 1;
        if target > self.m {
            return;
        }
        let mut nf = 0usize;
        for i in 0..self.n {
            let f = self.cap - self.cnt[i];
            if f > 0 {
                self.freei[nf] = i;
                self.freef[nf] = f; // cap <= 4 asserted at construction
                nf += 1;
            }
        }
        self.nfree = nf;
        if !self.feasible_to(&cov, j, target) {
            return;
        }
        // children (stack array, no allocation)
        let mut sc: [(i32, u32); 96] = [(0, 0); 96];
        let mut ns = 0usize;
        'outer: for i in 0..self.n {
            if self.cnt[i] >= self.cap {
                continue;
            }
            let p = self.ps[i];
            let r = j % p;
            if r == 0 {
                continue; // class 0 forbidden
            }
            // canonical form: reject if an earlier commit used a LARGER prime
            // at a position this class also covers
            for s in 0..self.ncom {
                if self.cp[s] != i && self.ps[self.cp[s]] > p && self.cj[s] % p == r {
                    self.psym += 1;
                    continue 'outer;
                }
            }
            let mut g: i32 = 0;
            for w in 0..self.nw {
                g += (self.masks[i][r][w] & !cov[w]).count_ones() as i32;
            }
            sc[ns] = (-g, i as u32);
            ns += 1;
        }
        if ns == 0 {
            return;
        }
        sc[..ns].sort_unstable();
        // subtree split for parallel runs
        let mut keep: [u32; 96] = [0; 96];
        let mut nk = 0usize;
        if depth == self.split_depth && self.nparts > 1 {
            for t in 0..ns {
                let idx = self.leafctr;
                self.leafctr += 1;
                if (idx as usize) % self.nparts == self.part {
                    keep[nk] = sc[t].1;
                    nk += 1;
                }
            }
        } else {
            for t in 0..ns {
                keep[nk] = sc[t].1;
                nk += 1;
            }
        }
        for t in 0..nk {
            let i = keep[t] as usize;
            let p = self.ps[i];
            let r = j % p;
            let mut nc: Bits = [0u64; MW];
            for w in 0..self.nw {
                nc[w] = cov[w] | self.masks[i][r][w];
            }
            self.cnt[i] += 1;
            self.cj[self.ncom] = j;
            self.cp[self.ncom] = i;
            self.ncom += 1;
            self.dfs(nc, depth + 1);
            self.ncom -= 1;
            self.cnt[i] -= 1;
            if self.aborted {
                return;
            }
        }
    }

    fn solution(&self) -> Vec<Vec<usize>> {
        let mut v = vec![Vec::new(); self.n];
        for s in 0..self.ncom {
            let i = self.cp[s];
            v[i].push(self.cj[s] % self.ps[i]);
        }
        v
    }
}

fn verify(ps: &[usize], cap: usize, sol: &[Vec<usize>], m: usize) -> bool {
    for i in 0..ps.len() {
        if sol[i].len() > cap {
            return false;
        }
        let mut s = sol[i].clone();
        s.sort_unstable();
        s.dedup();
        if s.len() != sol[i].len() {
            return false;
        }
        for &r in &sol[i] {
            if r == 0 || r >= ps[i] {
                return false;
            }
        }
    }
    for j in 1..=m {
        let mut ok = false;
        for i in 0..ps.len() {
            if sol[i].contains(&(j % ps[i])) {
                ok = true;
                break;
            }
        }
        if !ok {
            return false;
        }
    }
    true
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: jkcov6 <k> <z> [--lmax N] [--seed M] [--secs S] [--quiet] [--split D --part I --nparts N]");
        std::process::exit(2);
    }
    let k: usize = args[1].parse().unwrap();
    let z: usize = args[2].parse().unwrap();
    let mut lmax = 0usize;
    let mut seed = 0usize;
    let mut secs = f64::INFINITY;
    let mut quiet = false;
    let mut split = 0usize;
    let mut part = 0usize;
    let mut nparts = 1usize;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--lmax" => { lmax = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            "--secs" => { secs = args[i + 1].parse().unwrap(); i += 2; }
            "--split" => { split = args[i + 1].parse().unwrap(); i += 2; }
            "--part" => { part = args[i + 1].parse().unwrap(); i += 2; }
            "--nparts" => { nparts = args[i + 1].parse().unwrap(); i += 2; }
            "--quiet" => { quiet = true; i += 1; }
            _ => { eprintln!("unknown arg {}", args[i]); std::process::exit(2); }
        }
    }
    let ps_all = primes_upto(z);
    let d: usize = ps_all.iter().filter(|&&p| p <= k + 1).product::<usize>().max(1);
    if lmax == 0 {
        let npr = ps_all.iter().filter(|&&p| p > k + 1).count();
        lmax = std::cmp::max(32, 6 * k * npr * npr / 2 + 64);
        if lmax > 1500 { lmax = 1500; }
    }
    let t0 = Instant::now();
    let mut s = Solver::new(k, z, lmax);
    s.limit = secs;
    s.best = seed;
    s.split_depth = split;
    s.part = part;
    s.nparts = nparts;
    s.dfs([0u64; MW], 0);
    let el = t0.elapsed().as_secs_f64();
    let m = s.best;
    if m >= s.m {
        println!("WINDOW TOO SMALL (m={})", s.m);
        std::process::exit(3);
    }
    let ok = if s.bestsol.iter().all(|v| v.is_empty()) {
        seed > 0
    } else {
        verify(&s.ps, s.cap, &s.bestsol, m)
    };
    let jk = d * (m + 1);
    if quiet {
        println!("{} {} {} {} {} {:.3} {} {}", k, z, jk, m, s.nodes, el,
                 if s.aborted { "ABORTED" } else { "EXACT" }, ok);
    } else {
        println!("k={} z={}   j_k(P(z)) = {} = {} * ({} + 1)   m = {}", k, z, jk, d, m, m);
        println!("nodes={}  prune v0={} v1={} v2={} sym={}  {:.3} s  {}  witness_verify={}",
                 s.nodes, s.pv0, s.pv1, s.pv2, s.psym, el,
                 if s.aborted { "ABORTED (lower bound only)" } else { "EXACT" }, ok);
        let mut w = String::new();
        for i in 0..s.n {
            let mut v = s.bestsol[i].clone();
            v.sort_unstable();
            w.push_str(&format!("{}:{:?} ", s.ps[i], v));
        }
        println!("witness {}", w);
    }
}
