"""Q11-Q14 (spectral) and Q15-Q16 (bitwise)."""

import sys
from math import prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from core import open_mask_pair, walk_lengths_fast

OUT = []


def say(s=""):
    print(s)
    OUT.append(str(s))


def crt_freqs(gears, W):
    """alpha_g(a) = a * ((W/g)^{-1} mod g) mod g, for all a in Z_W."""
    a = np.arange(W)
    out = {}
    for g in gears:
        M = W // g
        c = pow(M % g, -1, g)
        out[g] = (a * c) % g
    return out


def per_gear_hat(g, alpha):
    """u_g^(alpha): (g-2)/g at alpha = 0, -(1 + omega^{2 alpha})/g otherwise."""
    w = np.exp(2j * np.pi * (2 * alpha) / g)
    val = -(1 + w) / g
    val = np.where(alpha == 0, (g - 2) / g, val)
    return val


def q11q12(sets):
    say("### Q11/Q12  the per-gear transform, the product, and full support")
    say()
    say("| gears | W | max |DFT - product| | min |O^(a)| | at a = 0 | "
        "shield coordinate real? |")
    say("|---|---|---|---|---|---|")
    for gears in sets:
        W = prod(gears)
        mask = open_mask_pair(gears).astype(float)
        hat = np.fft.fft(mask) / W          # (1/W) sum f(n) omega^{-an}
        al = crt_freqs(gears, W)
        pred = np.ones(W, dtype=complex)
        for g in gears:
            pred *= per_gear_hat(g, al[g])
        err = np.abs(hat - pred).max()
        mn = np.abs(hat).min()
        # shield coordinate: shift n -> n+1 makes the teeth +-1
        shifted = np.roll(mask, 1)          # indicator of n' = n + 1
        hat_s = np.fft.fft(shifted) / W
        # per gear the factor should be -(2/g) cos(2 pi alpha / g), real
        pred_s = np.ones(W, dtype=complex)
        for g in gears:
            f = -(2 / g) * np.cos(2 * np.pi * al[g] / g)
            f = np.where(al[g] == 0, (g - 2) / g, f)
            pred_s *= f
        err_s = np.abs(hat_s - pred_s).max()
        say(f"| {','.join(map(str,gears))} | {W:,} | {err:.2e} | {mn:.3e} | "
            f"{abs(hat[0]):.6f} | real, max error {err_s:.2e} |")
    say()


def q13(sets, Ls=(1, 2, 3, 4)):
    say("### Q13  the run indicator: a Dirichlet kernel per gear, and its zeros")
    say()
    say("| gears | L | starts = prod(g-2-L) | max |DFT - kernel product| | zeros of the run "
        "transform | predicted zeros gcd(L+2,g) > 1 |")
    say("|---|---|---|---|---|---|")
    for gears in sets:
        W = prod(gears)
        base = open_mask_pair(gears)
        for L in Ls:
            run = base.copy()
            for i in range(1, L):
                run &= np.roll(base, -i)
            cnt = int(run.sum())
            hat = np.fft.fft(run.astype(float)) / W
            al = crt_freqs(gears, W)
            pred = np.ones(W, dtype=complex)
            for g in gears:
                a = al[g]
                # transform of the complement of {0,-1,...,-(L+1)}
                s = np.zeros(W, dtype=complex)
                for t in range(0, L + 2):
                    s += np.exp(2j * np.pi * a * t / g)
                f = -s / g
                f = np.where(a == 0, (g - L - 2) / g, f)
                pred *= f
            err = np.abs(hat - pred).max()
            nz = int((np.abs(hat) < 1e-12).sum())
            predz = 0
            zero_any = np.zeros(W, dtype=bool)
            for g in gears:
                a = al[g]
                zero_any |= (a != 0) & ((a * (L + 2)) % g == 0)
            predz = int(zero_any.sum())
            say(f"| {','.join(map(str,gears))} | {L} | {cnt:,} = "
                f"{prod(g-2-L for g in gears):,} | {err:.2e} | {nz:,} | {predz:,} |")
    say()


def q15q16(sets):
    say("### Q15/Q16  the striker-parity bit (XOR) and the longest run of ones")
    say()
    say("| gears | m | W | #even | (W + prod(g-4))/2 | #odd | prod(g-4) = domino count |"
        " longest XOR run | F_top | equal? |")
    say("|---|---|---|---|---|---|---|---|---|---|")
    for gears in sets:
        W = prod(gears)
        idx = np.arange(W)
        par = np.zeros(W, dtype=np.int8)
        for g in gears:
            r = idx % g
            par ^= ((r == 0) | (r == g - 2)).astype(np.int8)
        ne = int((par == 0).sum())
        no = int((par == 1).sum())
        P = prod(g - 4 for g in gears)
        mask = open_mask_pair(gears)
        F = int(walk_lengths_fast(mask).max())
        # longest cyclic run of ones in par
        p2 = np.concatenate([par, par])
        best = cur = 0
        for v in p2:
            cur = cur + 1 if v == 1 else 0
            best = max(best, cur)
        best = min(best, W)
        m = len(gears)
        say(f"| {','.join(map(str,gears))} | {m} | {W:,} | {ne:,} | {(W+P)//2:,} | {no:,} | "
            f"{P:,} | {best} | {F} | {'YES' if best == F else 'no (' + str(F-best) + ' short)'} |")
    say()


if __name__ == "__main__":
    SMALL = [(7, 11, 13), (11, 13, 17), (13, 17, 19)]
    q11q12(SMALL + [(17, 19, 23), (7, 11, 13, 17)])
    q13(SMALL)
    q15q16([(7, 11, 13), (11, 13, 17), (13, 17, 19), (17, 19, 23), (19, 23, 29),
            (7, 11, 13, 17), (11, 13, 17, 19), (13, 17, 19, 23), (17, 19, 23, 29),
            (11, 13, 17, 19, 23)])
    with open("research/topmachine/r3/results/spectral.out", "w") as f:
        f.write("\n".join(OUT))
