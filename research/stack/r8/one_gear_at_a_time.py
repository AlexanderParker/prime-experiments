"""One machine, one gear at a time, step up.  Start from the high gears in reach.  Add the gears
in the machine's order: base gears (teeth fixed on column h), then band gears (teeth a, b placed
by E), then the high gears (teeth placed by E; a high gear strikes only by dividing a member).
At each addition: the gear's teeth, which survivors it takes (with the member and the cofactor),
and what remains.  Ends with the passing gears.

usage: uv run python research/stack/r8/one_gear_at_a_time.py 499
"""
import sys
from sympy import primerange, factorint
from pathlib import Path

def main():
    q = int(sys.argv[1])
    primes = list(primerange(2, q + 1)); base = []; P = 1
    for p in primes:
        if P * p <= q // 2: P *= p; base.append(p)
        else: break
    def alt(xs): return sum(x if i % 2 == 0 else -x for i, x in enumerate(xs))
    E = -1 + 2 * P * alt([p for p in primes if p not in base][::-1])
    r = int(q ** 0.5)
    gears = [p for p in primes if p >= 5]
    high = [p for p in primes if p > r]
    live = [h for h in high if q < E + 6 * h and E + 6 * h + 2 <= q * q]
    out = [__doc__.strip(), "", f"q = {q}, E = {E}, base {base}, band {[g for g in gears if g <= r and g not in base]}, high gears from {high[0]}", f"high gears in reach: {len(live)}: {live}", ""]
    for g in gears:
        inv = pow(6, -1, g); a, b = (-E * inv) % g, (-(E + 2) * inv) % g
        role = "base" if g in base else ("band" if g <= r else "high")
        if role == "base": assert (a, b) == (inv % g, (-inv) % g)
        taken = []
        for h in live:
            if h == g: continue
            L = E + 6 * h
            if h % g == a:
                taken.append((h, "left", L // g, factorint(L // g)))
            elif h % g == b:
                taken.append((h, "right", (L + 2) // g, factorint((L + 2) // g)))
        live = [h for h in live if h == g or h % g not in (a, b)]
        if not taken and role != "base" and not live: break
        desc = "; ".join(f"h = {h} ({m}, cofactor {c} = " + "*".join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(f.items())) + ")" for h, m, c, f in taken)
        out.append(f"gear {g:>3} ({role:<4}) teeth h = {a:>3}, {b:>3} mod {g}: takes {len(taken):>2} -> {len(live):>3} remain" + (f"   {desc}" if taken else ""))
        if not live: out.append("   nothing remains"); break
    out.append(""); out.append(f"passing gears: {live}")
    Path(f"research/stack/r8/results_one_gear_at_a_time_{q}.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
