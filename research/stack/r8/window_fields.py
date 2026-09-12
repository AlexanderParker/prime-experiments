"""Which fields kill in a machine's window, over many cycles; which rows kill in those fields;
periodicity of each field against the machine's cycle; the gears that killed per field in each
machine layer; and whether the window's twins reappear at the window's mirror.

Machine q (a prime): gears the primes up to q, window (q, q^2], cycle length q# (the primorial),
cycle c = the window shifted by (c - 1) q#, layer = window(q) minus window(p) = (p^2, q^2] for p
the prime before q, mirror of the number n about q#/2 = q# - n (a column (n, n+2) mirrors to the
column (q# - n - 2, q# - n)).
Fields as in the explorer and its twin (fields_twin.py): multiples, squares, products:j, higher:g
(kills whose smallest gear is g), higher1:g (g exactly once), lower:g (kills whose largest gear
is g), lower1:g. A kill = a composite n with n = 1 or 5 mod 6 (a twin-slot member). Rows of a
field on a kill = the gears dividing it that the field paints (all of them for multiples and
products, g and above for higher, g and below for lower).
Periodicity in the machine's cycle: a field is PERIODIC if its kill offsets (n - (c-1) q#) are the
same in every cycle checked, BECOMES PERIODIC if the same from some cycle on (at least the last
three cycles equal and not all), NEVER otherwise; NEVER KILLS if it has no kill in any cycle.
Usage: uv run python window_fields.py qmax cycles
"""
import sys
from collections import defaultdict
from sympy import primerange, prevprime, isprime, factorint


def fields_of(n, fac):
    """the fields that kill at n (n a twin-slot composite) with the rows each paints"""
    ps = sorted(fac); omega = sum(fac.values()); out = {}
    out['multiples'] = ps
    if len(ps) == 1 and fac[ps[0]] == 2: out['squares'] = ps
    out[f'products:{omega}'] = ps
    g, G = ps[0], ps[-1]
    out[f'higher:{g}'] = ps
    if fac[g] == 1: out[f'higher1:{g}'] = ps
    out[f'lower:{G}'] = ps
    if fac[G] == 1: out[f'lower1:{G}'] = ps
    return out


def main():
    qmax, C = int(sys.argv[1]), int(sys.argv[2])
    report = []
    summary_period = defaultdict(set)   # field kind -> set of verdicts
    per_layer_gears = {}
    mirror_rows = []
    for q in primerange(5, qmax + 1):
        p = prevprime(q); P = 1
        for r in primerange(2, q + 1): P *= r
        lo, hi = q, q * q
        kills_by_cycle = []   # per cycle: {field: {offset: rows}}
        for c in range(1, C + 1):
            shift = (c - 1) * P; d = {}
            for n in range(lo + 1 + shift, hi + 1 + shift):
                if n % 6 not in (1, 5) or isprime(n): continue
                fac = factorint(n)
                for f, rows in fields_of(n, fac).items():
                    d.setdefault(f, {})[n - shift] = rows
            kills_by_cycle.append(d)
        allf = sorted(set(f for d in kills_by_cycle for f in d), key=lambda s: (s.split(':')[0], int(s.split(':')[1]) if ':' in s else 0))
        report.append(f"\n## machine {q}: window ({q}, {q*q}], cycle {P}, {C} cycles")
        # which fields kill per cycle (appearance pattern)
        report.append("fields killing per cycle (kill count):")
        for c, d in enumerate(kills_by_cycle, 1):
            report.append(f"  cycle {c}: " + ', '.join(f"{f}({len(d[f])})" for f in allf if f in d))
        # periodicity
        report.append("periodicity in the machine's cycle:")
        for f in allf:
            pats = [tuple(sorted(d.get(f, {}))) for d in kills_by_cycle]
            if all(not x for x in pats): v = 'NEVER KILLS'
            elif all(x == pats[0] for x in pats): v = 'PERIODIC'
            elif len(pats) >= 3 and pats[-1] == pats[-2] == pats[-3] and pats[-1]:
                k = next(i for i in range(len(pats)) if all(x == pats[-1] for x in pats[i:])); v = f'BECOMES PERIODIC from cycle {k+1}'
            elif len(pats) >= 3 and pats[-1] == pats[-2] == pats[-3]:
                v = 'STOPS KILLING (kills only in cycles ' + ','.join(str(i+1) for i, x in enumerate(pats) if x) + ')'
            else: v = 'NEVER PERIODIC'
            summary_period[f.split(':')[0] + (':g<=q' if ':' in f and f.split(':')[0] in ('higher', 'higher1', 'lower', 'lower1') and int(f.split(':')[1]) <= q else (':g>q' if ':' in f and f.split(':')[0] in ('higher', 'higher1', 'lower', 'lower1') else ''))].add(v.split(' from')[0].split(' (')[0])
            rows = sorted(set(r for d in kills_by_cycle for rs in d.get(f, {}).values() for r in rs))
            report.append(f"  {f}: {v}; killer rows over all cycles: {rows[:16]}{' ...' if len(rows) > 16 else ''}")
        # gears that killed per field in the layer (p^2, q^2], cycle 1
        d1 = kills_by_cycle[0]; lay = {}
        for f in allf:
            rows = sorted(set(r for n, rs in d1.get(f, {}).items() if n > p * p for r in rs))
            if rows: lay[f] = rows
        per_layer_gears[q] = lay
        report.append(f"layer ({p*p}, {q*q}], cycle 1, gears that killed per field:")
        for f, rows in lay.items(): report.append(f"  {f}: {rows}")
        # mirror check: twins of the window, read from the top down, against the mirror q# - n
        tw = [n for n in range(hi, lo, -1) if n % 6 == 5 and isprime(n) and isprime(n + 2)]
        hits = [(n, P - n - 2) for n in tw if isprime(P - n - 2) and isprime(P - n)]
        mirror_rows.append((q, len(tw), len(hits), hits[:6]))
        report.append(f"mirror: window twins (left members, top down) {tw[:10]}{' ...' if len(tw) > 10 else ''}; twins again at the mirror q# - n: {len(hits)} of {len(tw)}: {hits[:6]}")
    out = '\n'.join(report)
    print(out)
    print("\n## summary of periodicity by field kind (verdicts seen over all machines)")
    for k, vs in sorted(summary_period.items()): print(f"  {k}: {sorted(vs)}")
    print("\n## mirror summary: machine q | window twins | twins again at the mirror")
    for q, nt, nh, ex in mirror_rows: print(f"  {q} | {nt} | {nh}")


if __name__ == "__main__":
    main()
