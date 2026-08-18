//! # gearsuite — prime and twin-prime gaps in the slot frame
//!
//! Every routine here is an instance of a law established (and in most cases
//! kernel-checked in `proofs/`) by this project. Nothing heuristic is used: no
//! probabilistic primality, no fitted constants, no trial division by numbers
//! that the laws prove cannot divide.
//!
//! ## The frame
//!
//! Every prime `> 3` is `6k-1` or `6k+1` for some `k >= 1`. Call `k` a **slot**;
//! it carries the pair `(6k-1, 6k+1)`. A **twin pair** is a slot whose two
//! members are both prime. Working in slots rather than integers is a free 6x
//! compression, and it is the frame in which the laws below are stated.
//!
//! ## The laws this crate is built from
//!
//! | law | statement | used by |
//! |---|---|---|
//! | tooth law | gear `q >= 5` blocks slot `k` iff `k = ±u (mod q)`, `6u = 1 (mod q)` | [`slot::teeth`] |
//! | closed-form tooth | `u' = round(q/6)`, i.e. `(q ± 1)/6` whichever is integral | [`slot::tooth_offset`] |
//! | slot cap | no gear divides both members (it would divide their difference, 2) | [`sieve`] two independent bitsets |
//! | onset at `q^2` | gear `q` blocks nothing below its own square | [`sieve::Segment::sieve`] start offsets |
//! | horizon | gears `< y` decide the window `(y, y^2)` exactly | [`sieve::gears_for`] |
//! | corridor mod 35 | gears 5 and 7 leave exactly 15 of 35 slot residues twin-eligible | [`corridor::EXPOSED`] |
//! | twin product | twin `(6m-1, 6m+1)` blocks slot `6m^2` (self-blindness) | [`slot::product_slot`] |
//! | merge law | `F(M+q)` is computable from the *old* machine alone | [`machine::f_next`] |
//! | literal cap | a literal chain has at most 6 members, every gear, forever | [`machine::LITERAL_CAP`] |
//!
//! ## Layout
//!
//! - [`slot`] — closed forms: teeth, shields, umbrellas, onsets. All O(1).
//! - [`corridor`] — the proven (5,7) corridor as a wheel.
//! - [`sieve`] — segmented slot sieve; the engine for everything at scale.
//! - [`machine`] — finite gear machines: openings, record gaps `F`, the merge law.

pub mod corridor;
pub mod machine;
pub mod sieve;
pub mod slot;

pub use sieve::{next_prime, next_twin_slot, prev_prime, PrimeGaps, TwinGaps};
pub use slot::{lo, hi, slot_of};
