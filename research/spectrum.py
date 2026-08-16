"""Frequency domain: the beat structure of the exposure signal.

The gears are periodic sources, so the natural place to look for their interference is the
frequency domain. Two signals over one period `P = prod q`:

    threat count   t(m) = number of gears threatening slot m      (a SUM of per-gear signals)
    exposure       E(m) = 1 if every gear exposes slot m, else 0  (a PRODUCT of them)

For gear `q` the threat indicator is `I_q(m) = 1` when `m = +/- u_q mod q`. Its transform over
`Z/q` is

    Ihat_q(k) = (2/q) cos(2 pi k u_q / q)

so the threat-count spectrum is supported exactly on the rationals `k/q` with those amplitudes.
Since `u_q = 6^{-1} mod q`, writing `k = 6r` gives amplitude `(2/q) cos(2 pi r / q)` at frequency
`6r/q` - the block period appears as a frequency multiplier, and the strongest lines sit at
`6/q, 12/q, 18/q, ...` with amplitude close to `2/q`.

The exposure signal is a product, `E(m) = prod_q e_q(m mod q)` with `e_q = 1 - I_q`, and each
factor depends on `m` only through `m mod q`. By CRT the transform therefore **factorises**:

    Ehat(k) = prod_q ehat_q(k_q),  where k_q = k mod q is the CRT component of the frequency

    ehat_q(0)     = (q - 2)/q
    ehat_q(k != 0) = -(2/q) cos(2 pi k u_q / q)

This is the beat structure in closed form. A frequency at which only one gear has a nonzero
component is a fundamental of that gear; a frequency at which several do is a combination tone -
a beat - and its amplitude is the *product* of the participating factors, so it is exponentially
small in the number of gears taking part.
"""

import sys
from math import cos, pi, prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slip_algebra import gears_upto, tooth


def signals(gear_set):
    """The threat-count and exposure signals over one period."""
    P = prod(gear_set)
    t = np.zeros(P, dtype=np.int32)
    for q in gear_set:
        u = tooth(q)
        for r in (u, q - u):
            t[r::q] += 1
    return t, (t == 0).astype(np.float64)


def crt_component(gear_set, q, k):
    """The frequency component belonging to gear `q` at global frequency `k`.

    Writing `m` through the CRT idempotents, `m = sum m_q e_q` with `e_q = (P/q) t_q` and
    `t_q = (P/q)^{-1} mod q`, gives `omega_P^{e_q} = omega_q^{t_q}`. So the component is
    `k t_q mod q`, not `k mod q` - the twist matters for every `k` where `t_q != 1`.
    """
    P = prod(gear_set)
    t_q = pow((P // q) % q, -1, q)
    return (k * t_q) % q


def ehat_factor(q, kq):
    """Per-gear transform factor at CRT component `kq`."""
    if kq % q == 0:
        return (q - 2) / q
    return -(2.0 / q) * cos(2 * pi * kq * tooth(q) / q)


def predicted_spectrum(gear_set, k):
    """Closed-form Ehat(k) as the product of per-gear factors."""
    return prod(ehat_factor(q, crt_component(gear_set, q, k)) for q in gear_set)


def participation(gear_set, k):
    """Which gears have a nonzero frequency component at k - the beat order."""
    return [q for q in gear_set if crt_component(gear_set, q, k) != 0]


if __name__ == "__main__":
    S = [5, 7, 11, 13]
    P = prod(S)
    t, E = signals(S)
    print(f"gears {S}, period {P}, exposed slots {int(E.sum())} = prod(q-2) = "
          f"{prod(q - 2 for q in S)}")

    F = np.fft.fft(E) / P

    print("\nverify the factorised spectrum against the FFT")
    print(f"  {'k':>8} {'beat order':>11} {'gears participating':>24} "
          f"{'FFT Ehat(k)':>14} {'closed form':>14} {'agree':>7}")
    tests = [0, 1, 5, 7, 11, 13, 35, 77, 143, 385, 1001, 5005 // 5, 2, 3, 100, 2503]
    worst = 0.0
    for k in tests:
        k %= P
        part = participation(S, k)
        got = F[k].real
        want = predicted_spectrum(S, k)
        worst = max(worst, abs(got - want))
        print(f"  {k:>8} {len(part):>11} {str(part):>24} {got:>14.9f} "
              f"{want:>14.9f} {str(abs(got - want) < 1e-12):>7}")

    allk = np.arange(P)
    pred = np.array([predicted_spectrum(S, int(k)) for k in allk])
    print(f"\n  max deviation over ALL {P} frequencies: "
          f"{np.abs(F.real - pred).max():.3e}  (imaginary part max "
          f"{np.abs(F.imag).max():.3e})")

    print("\namplitude hierarchy by beat order - how many gears take part")
    orders = np.array([len(participation(S, int(k))) for k in allk])
    for o in range(len(S) + 1):
        sel = orders == o
        if not sel.any():
            continue
        amp = np.abs(pred[sel])
        print(f"  order {o}: {int(sel.sum()):>5} frequencies, "
              f"max |Ehat| {amp.max():.3e}, mean |Ehat| {amp.mean():.3e}")
