//! The engine: a segmented sieve in the slot frame.
//!
//! One gear strikes exactly **two** arithmetic progressions — one per tooth —
//! and the two progressions are tracked in two independent bitsets because the
//! slot cap guarantees a gear never hits both members of a slot. Striking
//! starts at the gear's onset (`q^2`), which is both an optimisation and the
//! reason the gear never strikes its own pair.
//!
//! Costs, relative to a classical odd-only sieve of the same integer range:
//! the slot frame stores one bit per 3 integers instead of one per 2, and the
//! (5,7) corridor is applied by table rather than by striking.

use crate::corridor::{EXPOSED, HI_BLOCKED, LO_BLOCKED, WHEEL};

/// Can slot `k` carry a twin pair at all?
///
/// The corridor law: above gear 7, both members must avoid divisibility by 5
/// and by 7, so `k mod 35` must lie in the exposed set — 20 of every 35 slots
/// are ruled out with no sieving at all. Slot 1 is the one exception, since its
/// members *are* the gears 5 and 7 (the same `k >= 2` guard that the
/// kernel-checked form of this law carries).
#[inline(always)]
pub const fn twin_eligible(k: u64) -> bool {
    k <= 1 || EXPOSED[(k % WHEEL) as usize]
}
use crate::slot::{hi, lo, slot_of, teeth};

/// A sieved block of slots `[base, base + len)`.
///
/// `lo_comp[i]` is set when `6(base+i) - 1` is composite; `hi_comp[i]` when
/// `6(base+i) + 1` is composite.
pub struct Segment {
    pub base: u64,
    pub len: usize,
    lo_comp: Vec<bool>,
    hi_comp: Vec<bool>,
}

/// The gears needed to decide primality throughout `[2, limit]`.
///
/// This is the horizon law in its operational form: a member `m` is prime iff
/// no gear `q <= sqrt(m)` blocks its slot, so gears up to `sqrt(limit)` decide
/// the whole range and nothing above that can contribute.
pub fn gears_for(limit: u64) -> Vec<u64> {
    let bound = (limit as f64).sqrt() as u64 + 2;
    let mut sieve = vec![true; (bound + 1) as usize];
    let mut out = Vec::new();
    let mut p = 5u64; // gears start at 5; 2 and 3 are the frame, not gears
    let mut n = 2u64;
    while n <= bound {
        if sieve[n as usize] {
            let mut m = n * n;
            while m <= bound {
                sieve[m as usize] = false;
                m += n;
            }
        }
        n += 1;
    }
    while p <= bound {
        if sieve[p as usize] {
            out.push(p);
        }
        p += 1;
    }
    out
}

impl Segment {
    /// Sieve slots `[base, base + len)` using `gears`.
    ///
    /// `gears` must contain every prime `>= 5` up to `sqrt(6*(base+len)+1)`;
    /// [`gears_for`] produces exactly that set.
    pub fn sieve(base: u64, len: usize, gears: &[u64]) -> Segment {
        let mut lo_comp = vec![false; len];
        let mut hi_comp = vec![false; len];

        // Gears 5 and 7 are applied from the precomputed corridor pattern.
        for i in 0..len {
            let r = ((base + i as u64) % WHEEL) as usize;
            lo_comp[i] = LO_BLOCKED[r];
            hi_comp[i] = HI_BLOCKED[r];
        }
        // The pattern also strikes gears 5 and 7 themselves, which are both
        // members of slot 1. A gear never blocks its own pair, so restore it.
        if base <= 1 && 1 < base + len as u64 {
            let i = (1 - base) as usize;
            lo_comp[i] = false; // 5
            hi_comp[i] = false; // 7
        }

        for &q in gears {
            if q < 11 {
                continue; // handled by the wheel
            }
            let onset = slot_of(q * q);
            if onset >= base + len as u64 {
                break; // gears are ascending: nothing further can reach us
            }
            let (tl, tr) = teeth(q);
            strike(&mut lo_comp, base, len, q, tl, onset);
            strike(&mut hi_comp, base, len, q, tr, onset);
        }

        Segment { base, len, lo_comp, hi_comp }
    }

    /// Is the lower member of slot `base + i` prime?
    #[inline(always)]
    pub fn lo_prime(&self, i: usize) -> bool {
        !self.lo_comp[i]
    }

    /// Is the upper member of slot `base + i` prime?
    #[inline(always)]
    pub fn hi_prime(&self, i: usize) -> bool {
        !self.hi_comp[i]
    }

    /// Is slot `base + i` a twin pair?
    #[inline(always)]
    pub fn is_twin(&self, i: usize) -> bool {
        !self.lo_comp[i] && !self.hi_comp[i]
    }

    /// Twin slots in this segment, ascending.
    ///
    /// Only residues in the exposed set are examined — the corridor law says no
    /// others can qualify.
    pub fn twin_slots(&self) -> Vec<u64> {
        let mut out = Vec::new();
        for i in 0..self.len {
            let k = self.base + i as u64;
            if twin_eligible(k) && self.is_twin(i) {
                out.push(k);
            }
        }
        out
    }

    /// Primes in this segment, ascending.
    pub fn primes(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.len / 2);
        for i in 0..self.len {
            let k = self.base + i as u64;
            if self.lo_prime(i) {
                out.push(lo(k));
            }
            if self.hi_prime(i) {
                out.push(hi(k));
            }
        }
        out
    }
}

/// Strike one tooth: mark every slot `= t (mod q)` in `[base, base+len)` that is
/// at or above the gear's onset.
#[inline]
fn strike(bits: &mut [bool], base: u64, len: usize, q: u64, t: u64, onset: u64) {
    let start = base.max(onset);
    // first index >= start with (base + i) = t (mod q)
    let rem = start % q;
    let delta = if rem <= t { t - rem } else { q - rem + t };
    let mut k = start + delta;
    while k < base + len as u64 {
        bits[(k - base) as usize] = true;
        k += q;
    }
}

/// Default segment length in slots (about 1.5 MB of flags).
const SEG: usize = 1 << 18;

/// The next prime strictly greater than `n`.
///
/// Exact: primality is decided by the horizon law, not by a probabilistic test.
pub fn next_prime(n: u64) -> u64 {
    if n < 2 {
        return 2;
    }
    if n < 3 {
        return 3;
    }
    let mut base = slot_of(n.max(4)).max(1);
    loop {
        let limit = hi(base + SEG as u64);
        let gears = gears_for(limit);
        let seg = Segment::sieve(base, SEG, &gears);
        for i in 0..seg.len {
            let k = base + i as u64;
            if lo(k) > n && seg.lo_prime(i) {
                return lo(k);
            }
            if hi(k) > n && seg.hi_prime(i) {
                return hi(k);
            }
        }
        base += SEG as u64;
    }
}

/// The largest prime strictly less than `n`, or `None` when `n <= 2`.
pub fn prev_prime(n: u64) -> Option<u64> {
    if n <= 2 {
        return None;
    }
    if n <= 3 {
        return Some(2);
    }
    if n <= 5 {
        return Some(3);
    }
    let top = slot_of(n) + 1;
    let mut span = 256u64;
    loop {
        let base = top.saturating_sub(span).max(1);
        let len = (top - base) as usize;
        let gears = gears_for(hi(top));
        let seg = Segment::sieve(base, len, &gears);
        let mut best = None;
        for i in 0..len {
            let k = base + i as u64;
            if seg.lo_prime(i) && lo(k) < n {
                best = Some(lo(k));
            }
            if seg.hi_prime(i) && hi(k) < n {
                best = Some(hi(k));
            }
        }
        if let Some(p) = best {
            return Some(p);
        }
        if base == 1 {
            return Some(3);
        }
        span *= 4;
    }
}

/// The next twin slot strictly greater than slot `k`.
pub fn next_twin_slot(k: u64) -> u64 {
    let mut base = k + 1;
    loop {
        let limit = hi(base + SEG as u64);
        let gears = gears_for(limit);
        let seg = Segment::sieve(base, SEG, &gears);
        for i in 0..seg.len {
            let s = base + i as u64;
            if twin_eligible(s) && seg.is_twin(i) {
                return s;
            }
        }
        base += SEG as u64;
    }
}

/// Streaming prime gaps from a starting point.
pub struct PrimeGaps {
    seg: Segment,
    idx: usize,
    upper: bool,
    prev: Option<u64>,
    gears: Vec<u64>,
}

impl PrimeGaps {
    /// Gaps between consecutive primes, beginning at the first prime `>= from`.
    pub fn new(from: u64) -> PrimeGaps {
        let base = slot_of(from.max(5)).max(1);
        let gears = gears_for(hi(base + SEG as u64));
        PrimeGaps {
            seg: Segment::sieve(base, SEG, &gears),
            idx: 0,
            upper: false,
            prev: None,
            gears,
        }
    }

    fn advance(&mut self) -> u64 {
        loop {
            if self.idx >= self.seg.len {
                let base = self.seg.base + self.seg.len as u64;
                self.gears = gears_for(hi(base + SEG as u64));
                self.seg = Segment::sieve(base, SEG, &self.gears);
                self.idx = 0;
                self.upper = false;
            }
            let k = self.seg.base + self.idx as u64;
            if !self.upper {
                self.upper = true;
                if self.seg.lo_prime(self.idx) {
                    return lo(k);
                }
            } else {
                self.upper = false;
                self.idx += 1;
                if self.seg.hi_prime(self.idx - 1) {
                    return hi(k);
                }
            }
        }
    }
}

impl Iterator for PrimeGaps {
    /// `(prime, gap to the next prime)`
    type Item = (u64, u64);

    fn next(&mut self) -> Option<(u64, u64)> {
        let a = match self.prev {
            Some(p) => p,
            None => {
                let p = self.advance();
                p
            }
        };
        let b = self.advance();
        self.prev = Some(b);
        Some((a, b - a))
    }
}

/// Streaming twin-pair gaps, measured in slots.
pub struct TwinGaps {
    seg: Segment,
    idx: usize,
    prev: Option<u64>,
    gears: Vec<u64>,
}

impl TwinGaps {
    /// Twin gaps beginning at the first twin slot `>= from_slot`.
    pub fn new(from_slot: u64) -> TwinGaps {
        let base = from_slot.max(1);
        let gears = gears_for(hi(base + SEG as u64));
        TwinGaps {
            seg: Segment::sieve(base, SEG, &gears),
            idx: 0,
            prev: None,
            gears,
        }
    }

    fn advance(&mut self) -> u64 {
        loop {
            if self.idx >= self.seg.len {
                let base = self.seg.base + self.seg.len as u64;
                self.gears = gears_for(hi(base + SEG as u64));
                self.seg = Segment::sieve(base, SEG, &self.gears);
                self.idx = 0;
            }
            let k = self.seg.base + self.idx as u64;
            let i = self.idx;
            self.idx += 1;
            if twin_eligible(k) && self.seg.is_twin(i) {
                return k;
            }
        }
    }
}

impl Iterator for TwinGaps {
    /// `(twin slot, gap in slots to the next twin)`
    type Item = (u64, u64);

    fn next(&mut self) -> Option<(u64, u64)> {
        let a = match self.prev {
            Some(k) => k,
            None => self.advance(),
        };
        let b = self.advance();
        self.prev = Some(b);
        Some((a, b - a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        let mut d = 2;
        while d * d <= n {
            if n % d == 0 {
                return false;
            }
            d += 1;
        }
        true
    }

    #[test]
    fn segment_agrees_with_trial_division() {
        let gears = gears_for(hi(50_000));
        let seg = Segment::sieve(1, 50_000, &gears);
        for i in 0..seg.len {
            let k = 1 + i as u64;
            assert_eq!(seg.lo_prime(i), trial_prime(lo(k)), "lo mismatch at slot {k}");
            assert_eq!(seg.hi_prime(i), trial_prime(hi(k)), "hi mismatch at slot {k}");
        }
    }

    #[test]
    fn offset_segment_agrees() {
        let base = 1_000_000u64;
        let gears = gears_for(hi(base + 20_000));
        let seg = Segment::sieve(base, 20_000, &gears);
        for i in 0..seg.len {
            let k = base + i as u64;
            assert_eq!(seg.lo_prime(i), trial_prime(lo(k)), "lo mismatch at slot {k}");
            assert_eq!(seg.hi_prime(i), trial_prime(hi(k)), "hi mismatch at slot {k}");
        }
    }

    #[test]
    fn next_prime_matches_known_values() {
        assert_eq!(next_prime(0), 2);
        assert_eq!(next_prime(2), 3);
        assert_eq!(next_prime(3), 5);
        assert_eq!(next_prime(7), 11);
        assert_eq!(next_prime(89), 97);
        assert_eq!(next_prime(1_000_000), 1_000_003);
        assert_eq!(next_prime(1_000_000_000), 1_000_000_007);
    }

    #[test]
    fn prev_prime_matches_known_values() {
        assert_eq!(prev_prime(3), Some(2));
        assert_eq!(prev_prime(5), Some(3));
        assert_eq!(prev_prime(97), Some(89));
        assert_eq!(prev_prime(1_000_003), Some(999_983));
    }

    #[test]
    fn twin_slots_are_the_known_twins() {
        let gears = gears_for(hi(200));
        let seg = Segment::sieve(1, 200, &gears);
        let got: Vec<u64> = seg.twin_slots().into_iter().take(8).collect();
        // (5,7) (11,13) (17,19) (29,31) (41,43) (59,61) (71,73) (101,103)
        assert_eq!(got, vec![1, 2, 3, 5, 7, 10, 12, 17]);
    }

    #[test]
    fn prime_gap_stream_is_correct() {
        let got: Vec<(u64, u64)> = PrimeGaps::new(5).take(6).collect();
        assert_eq!(got, vec![(5, 2), (7, 4), (11, 2), (13, 4), (17, 2), (19, 4)]);
    }

    #[test]
    fn twin_gap_stream_is_correct() {
        let got: Vec<(u64, u64)> = TwinGaps::new(1).take(5).collect();
        // slots 1,2,3,5,7,10 -> gaps 1,1,2,2,3
        assert_eq!(got, vec![(1, 1), (2, 1), (3, 2), (5, 2), (7, 3)]);
    }

    #[test]
    fn twin_count_below_million() {
        // pi_2(10^6) = 8169 pairs counting (3,5). The slot frame cannot carry
        // (3,5) at all - 3 is divisible by 3, so it has no slot - and every
        // other twin pair does have one. The frame's count is therefore 8168,
        // and that difference is a property of the frame, not a miss.
        let gears = gears_for(1_000_001);
        let mut count = 0usize;
        let mut base = 1u64;
        let top = slot_of(1_000_000);
        while base <= top {
            let len = ((top - base + 1) as usize).min(1 << 16);
            let seg = Segment::sieve(base, len, &gears);
            for i in 0..len {
                let k = base + i as u64;
                if hi(k) <= 1_000_000 && seg.is_twin(i) {
                    count += 1;
                }
            }
            base += len as u64;
        }
        assert_eq!(count, 8168);
    }
}
