# The golden spectral gap and the T3 tooth-phase law

## 1. WHAT IT IS

Plain language: put the machine in frequency space. Each gear blocks two
residues (teeth) mod q, so its indicator is a two-term exponential sum, and
the whole machine's Fourier transform factorises over gears. Three exact
facts fall out. First, the entire spectrum is real and closed-form. Second,
for EVERY machine containing gear 5, the largest non-constant Fourier
coefficient, relative to the mean, is the golden ratio over three - a
machine-independent spectral gap, contributed by gear 5 alone. Third, at
local frequency 3 every gear's two teeth land on ADJACENT residues at the
antipode - in phase terms each gear is almost a single-point blocker there.

Precise form. A_q = Z_q - {u, -u}, u = 6^{-1} mod q. DFT: hat_q(0) = q-2 and

    hat_q(j) = -2 cos(2 pi j u / q)        (j != 0, always real),

and for the machine on Z_P (P = prod q) the global transform factorises:
hat(j) = prod_q hat_q(j (P/q)^{-1} mod q).

GOLDEN GAP: hat_5(2) = -2 cos(4 pi/5) = phi = (1+sqrt 5)/2 exactly, and for
every machine containing gear 5,

    max_{chi != trivial} |hat(chi)| / hat(trivial)  =  phi/3  =  0.53934...,

attained only by gear 5's local-frequency +-2 mode (all other gears
contribute max ratio 2 cos(pi/q)/(q-2) < phi/3, and mixing gears only
shrinks the product).

T3 LAW: 6u = 1 (mod q) forces 3u = (q+1)/2 (mod q), so the tripled teeth
{3u, -3u} = {(q+1)/2, (q-1)/2} are adjacent residues at the antipode, and
hat_q(3) = -2 cos(pi/q) -> -2, near-extremal for a 2-point set, for every
prime gear q >= 5. This is the Fourier avatar of the tooth law u' ~ q/6
(both teeth sit at +-60 degrees of phase) and of the round-19 identity
2u = 3^{-1} mod q.

## 2. WHY IT MIGHT BE NOVEL

The DFT of a CRT product set is classical, and two-point exponential sums are
trivial. What may be new is the specific structure: (a) the exact golden-
ratio value of the dominant mode and the machine-independent spectral gap
phi/3 for the twin-slot sieve - the spectral-gap form of the project's
repeated empirical finding that gear 5 controls all corridor phenomenology
(AP lemma, adjacent-gap exclusion, address pinning); (b) the T3 law as a
universal phase alignment of ALL gears at one frequency. Measured
consequence (round 20): the dominant oscillatory line of the log gap
histogram at machines 29/31 is the golden line at frequency 2/5, and
dividing by the closed-form arithmetic factor removes 99.6%+ of its power.

## 3. PROOF

Status: PROVED (elementary, lines above) + SCRIPT-VERIFIED; the T3 law is KERNEL-CHECKED
since round 20 (`LiteralCapTable.tripled_teeth_antipode` in proofs/LiteralCapTable.lean,
in the tooth-offset frame 6u' = q -+ 1: {3u', q - 3u'} = {(q-1)/2, (q+1)/2} exactly, for
every gear forever - standard three axioms).

research/machine_dft.py: per-gear closed form vs direct DFT for gears 5..43
(max deviation 8e-14); T3 law asserted for all primes 5 <= q <= 100000;
global factorisation vs FFT at machine 17 over all 85085 frequencies (max
deviation 4e-11); the phi/3 gap confirmed by full character enumeration at
machines 13 and 17; the golden-line collapse measured at machines 23/29/31.

## 4. IMPLICATIONS

Inside the project: exact power spectrum -> every autocorrelation-type census
has an exponential-sum form with known coefficients; the n-point correlation
closed forms (rounds 18-20) are its shadows. Any future large-sieve or
spectral bound on window counts starts from these exact coefficients. The
gap histogram's arithmetic wiggle is now identified with named spectral
lines. Outside: a clean, quotable exact constant (the golden ratio as the
dominant Fourier coefficient of the twin-residue sieve mod 5, and phi/3 as a
machine-independent spectral gap).

## 5. UNSOLVED QUESTIONS OR CONJECTURES IT TOUCHES

Spectral formulations of sieve extremes (Wall V): the gap phi/3 bounds how
fast exposure equidistributes; whether a genuine large-sieve inequality on
W_j (window counts) can be run from the exact spectrum is an open next
construct.

## 6. PRIOR-ART CHECK

Not yet checked (agent without web access). To check: exponential sums over
Jacobsthal/sieve residue systems; "golden ratio" appearances in DFTs of
2-point sets mod 5 (the value 2 cos(pi/5) = phi is classical); spectral gaps
of CRT product sets. The delta: the machine-specific statement and the
measured golden-line dominance of the gap histogram.
