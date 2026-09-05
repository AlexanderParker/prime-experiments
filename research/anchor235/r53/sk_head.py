"""r53 sk_head - the head-collision lemma, exactly.

At K = 4, 5, 6 the counting bound leaves exactly one tight case with b big gears and the
two-gear small part {5, 7}: capacity maxstrike(5,L) + maxstrike(7,L) + 2b equals L on the nose.
So a cover would need gear 5 and gear 7 to strike their individual maxima AND to strike
disjointly.  This script computes, exactly over all 35 phase pairs, the largest number of
columns of an L-run that {5, 7} can strike together, against the sum of their maxima.

Also the same for the K = 3 tight case (small part {5}, b = 2), and the hole-distance data
the hand proof of A(3) = 7 uses.
"""
import os

from sk_core import RESULTS, arc, strike_mask

LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def maxstrike(g, L):
    q, r = divmod(L, g)
    return 2 * q + (2 if r > arc(g) else (1 if r >= 1 else 0))


def best_union(S, L):
    best, arg = -1, None
    def rec(i, cov, phs):
        nonlocal best, arg
        if i == len(S):
            c = bin(cov).count("1")
            if c > best:
                best, arg = c, tuple(phs)
            return
        for ph in range(S[i]):
            rec(i + 1, cov | strike_mask(S[i], ph, L), phs + [ph])
    rec(0, 0, [])
    return best, arg


def main():
    os.makedirs(RESULTS, exist_ok=True)
    say("=" * 88)
    say("THE HEAD COLLISION: gear 5 and gear 7 cannot both be maximal and disjoint")
    say("=" * 88)
    say(f"{'L':>4} {'max(5)':>7} {'max(7)':>7} {'sum':>5} {'max |5 u 7|':>12} "
        f"{'deficit':>8} {'phases at the max':>18}")
    for L in (7, 15, 16, 21, 22, 27, 28, 36, 37, 44, 45):
        m5, m7 = maxstrike(5, L), maxstrike(7, L)
        u, arg = best_union([5, 7], L)
        say(f"{L:>4} {m5:>7} {m7:>7} {m5+m7:>5} {u:>12} {m5+m7-u:>8} {str(arg):>18}")
    say()
    say("The tight cases of the counting bound:")
    for K, L, b, S in ((3, 7, 2, [5]), (3, 7, 1, [5, 7]), (4, 16, 2, [5, 7]),
                       (5, 22, 3, [5, 7]), (6, 28, 4, [5, 7])):
        u, arg = best_union(S, L)
        cap = sum(maxstrike(g, L) for g in S)
        say(f"  K={K} L={L} b={b} S={S}: capacity {cap} + 2b={2*b} = {cap+2*b} vs L={L}; "
            f"true best union = {u}, so holes >= {L-u} against {2*b} the big gears can take "
            f"-> {'CLOSED' if L - u > 2 * b else 'needs the hole distances'}")
    say()
    say("=" * 88)
    say("THE A(3) = 7 CASE DATA (the hand proof's tables)")
    say("=" * 88)
    L = 7
    say("  gear 5, all phases, strike sets in a run of 7 columns:")
    for ph in range(5):
        m = strike_mask(5, ph, L)
        say(f"    phase {ph}: {[i for i in range(L) if m >> i & 1]}")
    say("  gear 7, all phases:")
    for ph in range(7):
        m = strike_mask(7, ph, L)
        say(f"    phase {ph}: {[i for i in range(L) if m >> i & 1]}")
    say("  the two-hole configurations of {5,7} at L = 7 (b = 1 case):")
    seen = set()
    for p5 in range(5):
        for p7 in range(7):
            cov = strike_mask(5, p5, L) | strike_mask(7, p7, L)
            H = tuple(i for i in range(L) if not (cov >> i & 1))
            if len(H) == 2:
                seen.add((H, H[1] - H[0]))
    for H, d in sorted(seen):
        say(f"    holes {list(H)} at distance {d}")
    say(f"  distances realised: {sorted({d for _H, d in seen})}")
    say("  the four-hole configurations of {5} alone at L = 7 (b = 2 case):")
    for p5 in range(5):
        cov = strike_mask(5, p5, L)
        H = [i for i in range(L) if not (cov >> i & 1)]
        ds = sorted({H[j] - H[i] for i in range(len(H)) for j in range(i + 1, len(H))})
        say(f"    phase {p5}: strikes {[i for i in range(L) if cov >> i & 1]}, "
            f"holes {H}, pair distances {ds}")
    say()
    say("=" * 88)
    say("MINIMUM HOLES of the five surviving small parts at K = 4, L = 16")
    say("=" * 88)
    for S in ([5, 7], [5, 7, 11], [5, 7, 13], [5, 11, 13], [5, 7, 11, 13]):
        u, arg = best_union(S, 16)
        say(f"  S = {str(S):>18}: best union {u:>2} of 16 columns, so at least "
            f"{16-u:>2} holes, phases {arg}")
    say()
    say("  and the two-hole minima that make the K = 4 cover work at L = 15:")
    for S in ([5, 7, 11],):
        u, arg = best_union(S, 15)
        say(f"  S = {str(S):>18}, L = 15: best union {u} of 15, so at least {15-u} holes")
    with open(os.path.join(RESULTS, "sk_head.txt"), "w") as f:
        f.write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()
