//! The (5,7) corridor, used as a wheel.
//!
//! Gears 5 and 7 between them block 20 of the 35 slot residues mod 35 in at
//! least one member, leaving exactly **15 twin-eligible residues** — the
//! *exposed set* `E`. This is the same object that the proof search uses for
//! its corridor laws, and (a fact worth stating, since it is what makes the
//! wheel exactly right) `|E| = 15 = (5-2)(7-2)` is the Hardy–Littlewood
//! arithmetic factor for these two gears.
//!
//! Two consequences are used here:
//!
//! * a twin pair can only sit at a slot whose residue mod 35 lies in `E`,
//!   so twin search skips 20/35 = 57% of slots outright, with no sieving;
//! * the pattern is fixed, so it is applied by table lookup rather than by
//!   striking gears 5 and 7 in the inner loop.
//!
//! Both members' composite-by-5-or-7 patterns are also precomputed, so a
//! segment starts from the correct residues instead of an empty bitset.

use crate::slot::teeth;

/// The wheel modulus: `5 * 7`.
pub const WHEEL: u64 = 35;

/// `true` at residues where **neither** member is divisible by 5 or 7.
///
/// These are the only residues at which a twin pair can occur above gear 7.
pub static EXPOSED: [bool; WHEEL as usize] = build_exposed();

/// `true` at residues where the **lower** member `6k-1` is divisible by 5 or 7.
pub static LO_BLOCKED: [bool; WHEEL as usize] = build_blocked(true);

/// `true` at residues where the **upper** member `6k+1` is divisible by 5 or 7.
pub static HI_BLOCKED: [bool; WHEEL as usize] = build_blocked(false);

const fn build_blocked(lower: bool) -> [bool; WHEEL as usize] {
    let mut out = [false; WHEEL as usize];
    let mut r = 0usize;
    while r < WHEEL as usize {
        let k = r as u64;
        // teeth() is const; unroll the two gears explicitly.
        let (l5, r5) = teeth(5);
        let (l7, r7) = teeth(7);
        let blocked = if lower {
            k % 5 == l5 || k % 7 == l7
        } else {
            k % 5 == r5 || k % 7 == r7
        };
        out[r] = blocked;
        r += 1;
    }
    out
}

const fn build_exposed() -> [bool; WHEEL as usize] {
    let lo = build_blocked(true);
    let hi = build_blocked(false);
    let mut out = [false; WHEEL as usize];
    let mut r = 0usize;
    while r < WHEEL as usize {
        out[r] = !lo[r] && !hi[r];
        r += 1;
    }
    out
}

/// The 15 twin-eligible residues mod 35, ascending.
pub fn exposed_residues() -> Vec<u64> {
    (0..WHEEL).filter(|&r| EXPOSED[r as usize]).collect()
}

/// Number of twin-eligible residues: 15.
pub fn exposed_count() -> usize {
    EXPOSED.iter().filter(|&&b| b).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slot::{hi, lo};

    #[test]
    fn exposed_set_has_fifteen_residues() {
        // |E| = (5-2)(7-2) = 15, the HL factor for gears 5 and 7.
        assert_eq!(exposed_count(), 15);
    }

    #[test]
    fn wheel_agrees_with_divisibility() {
        for k in 1..5_000u64 {
            let r = (k % WHEEL) as usize;
            let l_div = lo(k) % 5 == 0 || lo(k) % 7 == 0;
            let h_div = hi(k) % 5 == 0 || hi(k) % 7 == 0;
            assert_eq!(LO_BLOCKED[r], l_div, "lo pattern wrong at k = {k}");
            assert_eq!(HI_BLOCKED[r], h_div, "hi pattern wrong at k = {k}");
            assert_eq!(EXPOSED[r], !l_div && !h_div, "exposed pattern wrong at k = {k}");
        }
    }

    #[test]
    fn every_twin_above_seven_sits_in_the_exposed_set() {
        // Direct check: no twin pair with both members > 7 sits outside E.
        let is_prime = |n: u64| {
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
        };
        for k in 2..20_000u64 {
            if is_prime(lo(k)) && is_prime(hi(k)) {
                assert!(EXPOSED[(k % WHEEL) as usize], "twin at slot {k} outside E");
            }
        }
    }
}
