//! Finite gear machines, their record gaps `F`, and the merge law.
//!
//! A **machine** `M(y)` is the gear set `{5, 7, ..., y}`. Its **openings** are
//! the slots that no gear blocks, and they repeat with period `P = prod q`. The
//! record gap `F(M)` is the largest cyclic gap between consecutive openings.
//!
//! The interesting routine here is [`f_next`]: `F(M + q')` — the record gap of
//! the machine with one more gear — computed from the **old machine alone**,
//! with no scan of the new (and vastly longer) period. That is the merge law,
//! established by the proof search and verified there against every known value.

use crate::slot::teeth;

/// A literal chain has at most 6 members, for every gear, forever.
///
/// Proved (and kernel-checked as `literal_chain_le_six`) from the fact that a
/// literal chain's positions form a walk that must stay inside the exposed set
/// mod 35, whose maximal run depends only on `q mod 210`. The bound is sharp:
/// 6 is attained exactly at `q = 37, 53, 83, 127, 157, 173 (mod 210)`.
pub const LITERAL_CAP: usize = 6;

/// The gear set `{5, 7, ..., y}`.
pub fn gears_upto(y: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut n = 5u64;
    while n <= y {
        let mut prime = true;
        let mut d = 2;
        while d * d <= n {
            if n % d == 0 {
                prime = false;
                break;
            }
            d += 1;
        }
        if prime {
            out.push(n);
        }
        n += 1;
    }
    out
}

/// Period of a machine: the product of its gears.
pub fn period(gears: &[u64]) -> u64 {
    gears.iter().product()
}

/// The openings of a machine, over one full period, ascending.
///
/// Slot `k` is an opening when no gear blocks either member — that is, `k` is
/// congruent to neither tooth of any gear.
pub fn openings(gears: &[u64]) -> Vec<u64> {
    let p = period(gears);
    let mut alive = vec![true; p as usize];
    for &q in gears {
        let (tl, tr) = teeth(q);
        let mut k = tl;
        while k < p {
            alive[k as usize] = false;
            k += q;
        }
        let mut k = tr;
        while k < p {
            alive[k as usize] = false;
            k += q;
        }
    }
    (0..p).filter(|&k| alive[k as usize]).collect()
}

/// The record gap `F(M)`: the largest cyclic gap between consecutive openings.
pub fn f_max_gap(openings: &[u64], period: u64) -> u64 {
    if openings.len() < 2 {
        return period;
    }
    let mut best = 0;
    for w in openings.windows(2) {
        best = best.max(w[1] - w[0]);
    }
    // the wrap-around gap
    best.max(period - openings[openings.len() - 1] + openings[0])
}

/// The gap spectrum `F_j(M)` for `j = 1..=depth`: `F_j` is the largest sum of
/// `j` consecutive gaps.
///
/// `F_1` is `F`. The spectrum is what the merge law's depth argument is stated
/// against: a merged run of `k` deletions spans `k+1` consecutive gaps, so it
/// can never exceed `F_{k+1}`.
pub fn spectrum(openings: &[u64], period: u64, depth: usize) -> Vec<u64> {
    let n = openings.len();
    let mut out = Vec::with_capacity(depth);
    for j in 1..=depth {
        let mut best = 0u64;
        for i in 0..n {
            // sum of j consecutive gaps starting at opening i, cyclically
            let a = openings[i];
            let b_index = i + j;
            let b = if b_index < n {
                openings[b_index]
            } else {
                openings[b_index - n] + period
            };
            best = best.max(b - a);
        }
        out.push(best);
    }
    out
}

/// A merged run found by the merge law.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Merge {
    /// Index (into the old opening list) of the first deleted opening.
    pub start: usize,
    /// Number of openings deleted.
    pub kills: usize,
    /// Length of the resulting gap in the new machine.
    pub span: u64,
    /// `true` when at least one link is padded (two kills on the same tooth).
    pub padded: bool,
}

/// `F(M + q')` computed from the old machine alone — the **merge law**.
///
/// Adding gear `q'` deletes some openings; the new record gap is the largest
/// distance between two surviving openings across a run of deleted ones. A run
/// is realisable exactly when the residues work out, and because
/// `gcd(q', P) = 1` every residue assignment occurs somewhere in the new
/// period — so a run is realisable iff, walking along it, each successive
/// opening lands on one of the two teeth:
///
/// * gap `= 0 (mod q')` — both kills on the **same** tooth (a *padded* link);
/// * gap `= ±2u (mod q')` — kills on **opposite** teeth (a *literal* link).
///
/// Any other gap breaks the run. Returns the record gap and a witness.
pub fn f_next(openings: &[u64], period: u64, q: u64) -> (u64, Merge) {
    let n = openings.len();
    assert!(n > 1, "machine must have at least two openings");
    let (tl, tr) = teeth(q);
    let tooth = [tl, tr];

    // Extended view so runs may wrap the period boundary.
    let ext: Vec<u64> = openings
        .iter()
        .copied()
        .chain(openings.iter().map(|&o| o + period))
        .chain(openings.iter().map(|&o| o + 2 * period))
        .collect();

    let mut best = 0u64;
    let mut witness = Merge { start: 0, kills: 0, span: 0, padded: false };

    for i in 0..n {
        // The empty run: an ordinary old gap survives untouched.
        let plain = ext[i + 1] - ext[i];
        if plain > best {
            best = plain;
            witness = Merge { start: i, kills: 0, span: plain, padded: false };
        }

        // Runs of deletions starting at opening i+1, on either tooth.
        for &t0 in &tooth {
            let mut r = t0;
            let mut padded = false;
            let mut kills = 0usize;
            // extend while the next opening also lands on a tooth
            while kills < n {
                let idx = i + 1 + kills;
                if kills > 0 {
                    let step = ext[idx] - ext[idx - 1];
                    let nr = (r + step % q) % q;
                    if nr == r {
                        padded = true;
                    } else if nr != tl && nr != tr {
                        break;
                    }
                    r = nr;
                }
                kills += 1;
                let span = ext[i + 1 + kills] - ext[i];
                if span > best {
                    best = span;
                    witness = Merge { start: i + 1, kills, span, padded };
                }
            }
        }
    }
    (best, witness)
}

/// Build the ladder of record gaps `F(M(y))` for successive machines.
///
/// Returns `(y, F_slot, F_adjacent)` per step. `F_adjacent = 3 * F_slot` is the
/// corpus's frame (`F(2,y) = 6, 15, 21, 33, ...`); the factor 3 is the frame
/// conversion, not a result.
pub fn ladder(max_y: u64) -> Vec<(u64, u64, u64)> {
    let mut out = Vec::new();
    let all = gears_upto(max_y);
    for i in 1..=all.len() {
        let gs = &all[..i];
        let p = period(gs);
        let ops = openings(gs);
        let f = f_max_gap(&ops, p);
        out.push((gs[i - 1], f, 3 * f));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_five_seven_has_the_corridor_openings() {
        let gs = gears_upto(7);
        assert_eq!(period(&gs), 35);
        // The openings of {5,7} are exactly the exposed residues mod 35.
        let ops = openings(&gs);
        assert_eq!(ops.len(), 15, "|E| = 15");
        let expected = crate::corridor::exposed_residues();
        assert_eq!(ops, expected);
    }

    #[test]
    fn f_ladder_matches_the_known_values() {
        // Slot frame: 2, 5, 7, 11, 18, 25 ... adjacent frame: 6, 15, 21, 33, 54, 75
        let got = ladder(19);
        let f_slot: Vec<u64> = got.iter().map(|t| t.1).collect();
        let f_adj: Vec<u64> = got.iter().map(|t| t.2).collect();
        assert_eq!(f_slot, vec![2, 5, 7, 11, 18, 25]);
        assert_eq!(f_adj, vec![6, 15, 21, 33, 54, 75]);
    }

    #[test]
    fn merge_law_predicts_the_next_record_from_the_old_machine() {
        // Each step is computed from the previous machine alone.
        let cases: &[(u64, u64, u64)] = &[
            // (machine y, next gear, known F of the next machine)
            (7, 11, 7),
            (11, 13, 11),
            (13, 17, 18),
            (17, 19, 25),
            (19, 23, 34),
        ];
        for &(y, q, expect) in cases {
            let gs = gears_upto(y);
            let p = period(&gs);
            let ops = openings(&gs);
            let (f, w) = f_next(&ops, p, q);
            assert_eq!(f, expect, "merge law wrong at {y} -> {q} (witness {w:?})");
        }
    }

    #[test]
    fn merge_law_agrees_with_direct_construction() {
        // Cross-check the law against brute-force construction of the new machine.
        for (y, q) in [(7u64, 11u64), (11, 13), (13, 17), (17, 19)] {
            let old = gears_upto(y);
            let ops = openings(&old);
            let (predicted, _) = f_next(&ops, period(&old), q);

            let mut new_gears = old.clone();
            new_gears.push(q);
            let direct = f_max_gap(&openings(&new_gears), period(&new_gears));
            assert_eq!(predicted, direct, "merge law disagrees with construction at {y} -> {q}");
        }
    }

    #[test]
    fn spectrum_starts_with_f_and_increases() {
        let gs = gears_upto(13);
        let ops = openings(&gs);
        let p = period(&gs);
        let sp = spectrum(&ops, p, 6);
        assert_eq!(sp[0], f_max_gap(&ops, p));
        assert_eq!(sp[0], 11);
        for w in sp.windows(2) {
            assert!(w[1] >= w[0], "spectrum must be non-decreasing");
        }
    }

    #[test]
    fn literal_chains_respect_the_cap() {
        // No purely literal run may exceed 6 kills, at any gear.
        for (y, q) in [(11u64, 13u64), (13, 17), (17, 19), (19, 23)] {
            let gs = gears_upto(y);
            let ops = openings(&gs);
            let p = period(&gs);
            let (_, w) = f_next(&ops, p, q);
            if !w.padded {
                assert!(w.kills <= LITERAL_CAP, "literal cap violated at {y} -> {q}: {w:?}");
            }
        }
    }
}
