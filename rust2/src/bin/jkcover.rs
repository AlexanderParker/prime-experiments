// jkcover.rs -- exact maximal-covering search for the k-class Jacobsthal family.
//
// HARVESTER lane, round 28.
//
//   j_k(P(z)) - 1  =  the longest interval coverable by choosing, at each prime
//                     p <= z, a set S_p of residue classes mod p with
//                     |S_p| <= min(k, p-1).
//
// k = 1 is the ordinary Jacobsthal function (A048669), k = 2 is Ziller-Morack's
// paired h_2.  This binary maximises the coverable run exactly.
//
// SEARCH.  Branch on "which prime covers the leftmost uncovered position".
// Every uncovered position has the SAME option set (a prime whose class already
// contains it would have covered it), so the leftmost position is as good a
// branching variable as any and gives the search a prefix structure.  Depth is
// bounded by sum_p min(k, p-1) -- 41 for k = 2, z = 73 -- because each branch
// commits one class.
//
// PRUNING.  At a node with leftmost uncovered position j and incumbent best L*,
// let U = uncovered positions in [j, L*+1).  Two capacity bounds:
//   (v1) sum_p f_p * ceil(M/p) >= |U|,  M = L*+1-j          (cheap, O(#primes))
//   (v2) sum_p (the f_p largest |U cap (r mod p)|) >= |U|    (exact per-prime)
// v2 is the LP-style relaxation of the residual set-cover and is the workhorse.
//
// Usage:
//   jkcover <k> <z> [--lmax N] [--seed L] [--secs S] [--quiet]
//     --seed L : start with incumbent L (use the known answer to time the
//                infeasibility half alone).
//     --secs S : abort after S seconds (reports ABORTED; the incumbent is then
//                only a lower bound).

use std::env;
use std::time::Instant;

const MW: usize = 32; // 2048 positions max
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
    caps: Vec<usize>,
    n: usize,
    lmax: usize,
    nw: usize,
    masks: Vec<Vec<Bits>>,
    cnt: Vec<usize>,
    used: Vec<Vec<usize>>,
    best: usize,
    bestsol: Vec<Vec<usize>>,
    nodes: u64,
    prunes_v1: u64,
    prunes_v2: u64,
    t0: Instant,
    limit: f64,
    aborted: bool,
    // scratch
    unc: Vec<usize>,
    hist: Vec<u32>,
    order: Vec<usize>,
}

impl Solver {
    fn new(k: usize, z: usize, lmax: usize) -> Solver {
        let ps = primes_upto(z);
        let caps: Vec<usize> = ps.iter().map(|&p| std::cmp::min(k, p - 1)).collect();
        let n = ps.len();
        let nw = (lmax + 63) / 64;
        assert!(nw <= MW, "lmax too large");
        let mut masks = Vec::with_capacity(n);
        for &p in &ps {
            let mut mm = Vec::with_capacity(p);
            for r in 0..p {
                let mut b: Bits = [0u64; MW];
                let mut j = r;
                while j < lmax {
                    b[j >> 6] |= 1u64 << (j & 63);
                    j += p;
                }
                mm.push(b);
            }
            masks.push(mm);
        }
        let maxp = *ps.last().unwrap_or(&2);
        Solver {
            ps,
            caps,
            n,
            lmax,
            nw,
            masks,
            cnt: vec![0; n],
            used: vec![Vec::new(); n],
            best: 0,
            bestsol: vec![Vec::new(); n],
            nodes: 0,
            prunes_v1: 0,
            prunes_v2: 0,
            t0: Instant::now(),
            limit: f64::INFINITY,
            aborted: false,
            unc: Vec::with_capacity(lmax),
            hist: vec![0; maxp + 1],
            order: Vec::with_capacity(n),
        }
    }

    #[inline]
    fn first_unset(&self, cov: &Bits) -> usize {
        for w in 0..self.nw {
            let v = !cov[w];
            if v != 0 {
                let j = (w << 6) + v.trailing_zeros() as usize;
                if j < self.lmax {
                    return j;
                }
                return self.lmax;
            }
        }
        self.lmax
    }

    // Necessary condition for reaching `target` (i.e. covering [0, target)).
    fn feasible_to(&mut self, cov: &Bits, j: usize, target: usize) -> bool {
        // collect uncovered positions in [j, target)
        self.unc.clear();
        let mut x = j;
        while x < target {
            if (cov[x >> 6] >> (x & 63)) & 1 == 0 {
                self.unc.push(x);
            }
            x += 1;
        }
        let u = self.unc.len();
        if u == 0 {
            return true;
        }
        // v1 -- cheap
        let m = target - j;
        let mut cap: usize = 0;
        for i in 0..self.n {
            let f = self.caps[i] - self.cnt[i];
            if f > 0 {
                cap += f * ((m + self.ps[i] - 1) / self.ps[i]);
                if cap >= u {
                    break;
                }
            }
        }
        if cap < u {
            self.prunes_v1 += 1;
            return false;
        }
        // v2 -- per-prime exact class counts on the residual set (small primes
        // first, so the running total reaches u and exits early in the common
        // non-pruning case)
        let mut total: usize = 0;
        for i in 0..self.n {
            let f = self.caps[i] - self.cnt[i];
            if f == 0 {
                continue;
            }
            let p = self.ps[i];
            for r in 0..p {
                self.hist[r] = 0;
            }
            for &y in &self.unc {
                self.hist[y % p] += 1;
            }
            // sum of the f largest
            let mut top: [u32; 8] = [0; 8];
            let ff = if f > 8 { 8 } else { f };
            for r in 0..p {
                let v = self.hist[r];
                if v > top[ff - 1] {
                    let mut q = ff - 1;
                    while q > 0 && top[q - 1] < v {
                        top[q] = top[q - 1];
                        q -= 1;
                    }
                    top[q] = v;
                }
            }
            for q in 0..ff {
                total += top[q] as usize;
            }
            if total >= u {
                return true;
            }
        }
        if total < u {
            self.prunes_v2 += 1;
            return false;
        }
        true
    }

    fn dfs(&mut self, cov: Bits, depth: usize) {
        self.nodes += 1;
        if self.nodes & 0xFFFFF == 0 && self.t0.elapsed().as_secs_f64() > self.limit {
            self.aborted = true;
        }
        if self.aborted {
            return;
        }
        let j = self.first_unset(&cov);
        if j >= self.lmax {
            // covered the whole window -- lmax was too small
            self.best = self.lmax;
            self.bestsol = self.used.clone();
            return;
        }
        if j > self.best {
            self.best = j;
            self.bestsol = self.used.clone();
        }
        let target = self.best + 1;
        if target > self.lmax {
            return;
        }
        if !self.feasible_to(&cov, j, target) {
            return;
        }
        // children: which prime covers j
        self.order.clear();
        for i in 0..self.n {
            if self.cnt[i] < self.caps[i] {
                self.order.push(i);
            }
        }
        if self.order.is_empty() {
            return;
        }
        // order by coverage gain, descending (find good incumbents fast)
        let mut scored: Vec<(i64, usize)> = Vec::with_capacity(self.order.len());
        for &i in &self.order {
            let r = j % self.ps[i];
            let mut g: i64 = 0;
            for w in 0..self.nw {
                g += (self.masks[i][r][w] & !cov[w]).count_ones() as i64;
            }
            scored.push((-g, i));
        }
        scored.sort_unstable();
        for &(_, i) in &scored {
            let r = j % self.ps[i];
            let mut nc: Bits = [0u64; MW];
            for w in 0..self.nw {
                nc[w] = cov[w] | self.masks[i][r][w];
            }
            self.cnt[i] += 1;
            self.used[i].push(r);
            self.dfs(nc, depth + 1);
            self.used[i].pop();
            self.cnt[i] -= 1;
            if self.aborted {
                return;
            }
        }
    }
}

fn verify(ps: &[usize], caps: &[usize], sol: &[Vec<usize>], l: usize) -> bool {
    for i in 0..ps.len() {
        if sol[i].len() > caps[i] {
            return false;
        }
        let mut s = sol[i].clone();
        s.sort_unstable();
        s.dedup();
        if s.len() != sol[i].len() {
            return false;
        }
        for &r in &sol[i] {
            if r >= ps[i] {
                return false;
            }
        }
    }
    for j in 0..l {
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
        eprintln!("usage: jkcover <k> <z> [--lmax N] [--seed L] [--secs S] [--quiet]");
        std::process::exit(2);
    }
    let k: usize = args[1].parse().unwrap();
    let z: usize = args[2].parse().unwrap();
    let mut lmax = 0usize;
    let mut seed = 0usize;
    let mut secs = f64::INFINITY;
    let mut quiet = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--lmax" => {
                lmax = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--seed" => {
                seed = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--secs" => {
                secs = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--quiet" => {
                quiet = true;
                i += 1;
            }
            _ => {
                eprintln!("unknown arg {}", args[i]);
                std::process::exit(2);
            }
        }
    }
    if lmax == 0 {
        // generous default window
        lmax = 64;
        let ps = primes_upto(z);
        let mut cap = 0usize;
        for &p in &ps {
            cap += std::cmp::min(k, p - 1) * 4;
        }
        lmax = std::cmp::max(lmax, cap * 32);
        if lmax > 2040 {
            lmax = 2040;
        }
    }
    let t0 = Instant::now();
    let mut s = Solver::new(k, z, lmax);
    s.limit = secs;
    s.best = seed;
    s.dfs([0u64; MW], 0);
    let el = t0.elapsed().as_secs_f64();
    let l = s.best;
    let ok = if s.bestsol.iter().all(|v| v.is_empty()) {
        seed > 0
    } else {
        verify(&s.ps, &s.caps, &s.bestsol, l)
    };
    if l >= s.lmax {
        println!("WINDOW TOO SMALL (lmax={})", s.lmax);
        std::process::exit(3);
    }
    if quiet {
        println!(
            "{} {} {} {} {} {:.3} {}",
            k,
            z,
            l + 1,
            l,
            s.nodes,
            el,
            if s.aborted { "ABORTED" } else { "EXACT" }
        );
    } else {
        println!("k={} z={}  j_k(P(z)) = {}   L = {}", k, z, l + 1, l);
        println!(
            "nodes={}  prunes v1={} v2={}  {:.3} s  {}  witness_verify={}",
            s.nodes,
            s.prunes_v1,
            s.prunes_v2,
            el,
            if s.aborted { "ABORTED (lower bound only)" } else { "EXACT" },
            ok
        );
        let mut w = String::new();
        for i in 0..s.n {
            let mut v = s.bestsol[i].clone();
            v.sort_unstable();
            w.push_str(&format!("{}:{:?} ", s.ps[i], v));
        }
        println!("witness {}", w);
    }
}
