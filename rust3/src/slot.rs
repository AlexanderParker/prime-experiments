//! Closed forms. Every function here is O(1) arithmetic — no search, no loop
//! over candidates. These are the "one gear" laws the project established
//! before any machinery was built on top of them.

/// Lower member of slot `k`: `6k - 1`.
#[inline(always)]
pub const fn lo(k: u64) -> u64 {
    6 * k - 1
}

/// Upper member of slot `k`: `6k + 1`.
#[inline(always)]
pub const fn hi(k: u64) -> u64 {
    6 * k + 1
}

/// The slot carrying `m`, for any `m` coprime to 6.
///
/// Both members of a slot map back to it: `slot_of(6k-1) = slot_of(6k+1) = k`.
/// This is the `(m + 1) / 6` placement law (`Placement.slotOf`).
#[inline(always)]
pub const fn slot_of(m: u64) -> u64 {
    (m + 1) / 6
}

/// `true` when `m` lies in the slot frame at all (`m` coprime to 6).
///
/// The frame covers every prime except 2 and 3.
#[inline(always)]
pub const fn in_frame(m: u64) -> bool {
    m % 2 != 0 && m % 3 != 0
}

/// The tooth offset `u'` of gear `q`: the smaller of the two blocked residues.
///
/// **Closed form, no modular inversion.** `6u = 1 (mod q)` has the solution
/// `u = (q+1)/6` when `q = 5 (mod 6)` and `u = q - (q-1)/6` when `q = 1 (mod 6)`.
/// In both cases the *smaller* tooth is `round(q/6) = (q + 1) / 6` in integer
/// arithmetic, because exactly one of `q-1`, `q+1` is divisible by 6.
///
/// `u'` is also the slot of the gear's own pair — the self-blocking law.
#[inline(always)]
pub const fn tooth_offset(q: u64) -> u64 {
    (q + 1) / 6
}

/// The two residues mod `q` that gear `q` blocks, as `(left, right)`.
///
/// `left` is the residue where `q | 6k - 1` (the gear kills the lower member);
/// `right` is where `q | 6k + 1` (it kills the upper member). They sum to `q` —
/// the reflection law — and they are distinct for every `q >= 5`, which is the
/// slot cap: one gear can never take both members of one slot.
#[inline(always)]
pub const fn teeth(q: u64) -> (u64, u64) {
    // 6k = 1 (mod q) kills the lower member; 6k = -1 (mod q) kills the upper.
    let u = tooth_offset(q);
    if q % 6 == 5 {
        // u = (q+1)/6 satisfies 6u = q + 1 = 1, so u kills the LOWER member.
        (u, q - u)
    } else {
        // q = 1 (mod 6): 6u = q + 1 = 2 ... the integral solution flips sides.
        (q - u, u)
    }
}

/// The shield of gear `q`: the slot at which the gear's two teeth are
/// symmetric about the origin — slot `k = 0 (mod q)`, blocked by neither tooth.
///
/// Kept as a named function because the umbrella laws are stated relative to it.
#[inline(always)]
pub const fn shield(_q: u64) -> u64 {
    0
}

/// Length of the **short umbrella** of gear `q`: the run of consecutive slots
/// centred on the shield that the gear does not block.
///
/// The teeth sit at `±u'`, so the unblocked run through 0 spans `2u' - 1` slots.
#[inline(always)]
pub const fn short_umbrella(q: u64) -> u64 {
    2 * tooth_offset(q) - 1
}

/// Length of the **long umbrella** of gear `q`: the complementary unblocked run,
/// wrapping the far side of the wheel.
#[inline(always)]
pub const fn long_umbrella(q: u64) -> u64 {
    q - 2 * tooth_offset(q) - 1
}

/// The first slot gear `q` can block: the onset law, `q` blocks nothing below
/// `q^2`.
///
/// Returned as the slot carrying `q^2`. Below this the gear's residues are
/// occupied only by multiples whose cofactor is smaller than `q`, which some
/// smaller gear has already claimed.
#[inline(always)]
pub const fn onset_slot(q: u64) -> u64 {
    slot_of(q * q)
}

/// The slot blocked by the product of a twin pair `(6m-1, 6m+1)`.
///
/// `(6m-1)(6m+1) = 36m^2 - 1 = 6(6m^2) - 1`, so the product is the *lower*
/// member of slot `6m^2`. This is the self-blindness law: a twin pair, acting
/// as a pair of gears, blocks exactly one slot jointly, and it is this one.
#[inline(always)]
pub const fn product_slot(m: u64) -> u64 {
    6 * m * m
}

/// Separation between a gear's two teeth measured in the k-frame: `3^-1 (mod q)`.
///
/// The correct frame for chain arithmetic — measuring tooth separation in
/// adjacent slots instead of this is the "frame trap" that produced a spurious
/// `prod(q-4)` law early in the project.
pub fn tooth_separation(q: u64) -> u64 {
    // 3 * s = 1 (mod q). For q coprime to 3, s = (q+1)/3 or (2q+1)/3.
    match q % 3 {
        1 => (2 * q + 1) / 3,
        2 => (q + 1) / 3,
        _ => 0, // q = 3 has no inverse; not a gear in this frame
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_roundtrip() {
        for k in 1..10_000u64 {
            assert_eq!(slot_of(lo(k)), k);
            assert_eq!(slot_of(hi(k)), k);
        }
    }

    #[test]
    fn teeth_are_the_blocked_residues() {
        // The defining property, checked directly against divisibility.
        for q in [5u64, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 97, 101] {
            let (l, r) = teeth(q);
            assert_ne!(l, r, "slot cap: teeth must be distinct for q = {q}");
            assert_eq!(l + r, q, "reflection law: teeth sum to the modulus");
            for k in 1..(3 * q) {
                assert_eq!(lo(k) % q == 0, k % q == l, "left tooth wrong at q={q}, k={k}");
                assert_eq!(hi(k) % q == 0, k % q == r, "right tooth wrong at q={q}, k={k}");
            }
        }
    }

    #[test]
    fn tooth_offset_is_round_q_over_6() {
        for q in [5u64, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53] {
            let u = tooth_offset(q);
            let (l, r) = teeth(q);
            assert_eq!(u, l.min(r), "u' is the smaller tooth");
            // round(q/6) with integer arithmetic
            assert_eq!(u, (q + 3) / 6, "closed form disagrees at q = {q}");
        }
    }

    #[test]
    fn umbrellas_partition_the_wheel() {
        for q in [5u64, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
            // two teeth + two umbrellas = the whole wheel
            assert_eq!(short_umbrella(q) + long_umbrella(q) + 2, q, "q = {q}");
        }
    }

    #[test]
    fn product_slot_is_the_twin_product() {
        for m in [1u64, 2, 3, 5, 7, 10, 12, 17, 100] {
            assert_eq!(lo(product_slot(m)), lo(m) * hi(m));
        }
    }

    #[test]
    fn onset_is_the_square() {
        for q in [5u64, 7, 11, 13, 17, 19, 23] {
            let k = onset_slot(q);
            assert!(lo(k) == q * q || hi(k) == q * q, "q^2 must be a member of its onset slot");
        }
    }

    #[test]
    fn tooth_separation_inverts_three() {
        for q in [5u64, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43] {
            assert_eq!(3 * tooth_separation(q) % q, 1, "q = {q}");
        }
    }
}
