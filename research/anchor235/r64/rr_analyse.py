"""rr_analyse.py -- the tables of branch record_2run.md, from the profile scans.

Reads results/prof_*.json (rr_profile.py) and prints:
  * the headline row per machine: F, F_2, n1(F), Sig(F), D_rec, the F_2 pair, D_top;
  * the top band: v, m(v), n1(v), N(v), Sig(v), and the rarity null;
  * isolation of the top of the spectrum;
  * the exceptionless checks of the branch.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")


def load():
    prof = {}
    for f in sorted(os.listdir(OUT)):
        if f.startswith("prof_") and f.endswith(".json"):
            prof.update(json.load(open(os.path.join(OUT, f))))
    return {int(k): {kk: (({int(a): b for a, b in vv.items()}
                           if isinstance(vv, dict) and kk in
                           ("spec", "n1", "N", "Sig", "wit") else vv))
                     for kk, vv in v.items()} for k, v in prof.items()}


def rarity_null(spec, Ntot, draws):
    """largest r with draws * P(a uniformly random gap is >= r) >= 1."""
    sizes = sorted(spec)
    best = 0
    for r in sizes:
        tail = sum(spec[s] for s in sizes if s >= r)
        if draws * tail >= Ntot:
            best = r
    return best


def main():
    prof = load()
    ms = sorted(prof)
    print("\n### headline: the record as a 2-run\n")
    print("| M | F | F_2 | m(F) | n1(F) | F+n1(F) | D_rec | F_2 pair(s) | max member / F | D_top |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for y in ms:
        p = prof[y]
        F, F2 = p["F"], p["F2"]
        n1 = p["n1"]
        spec = p["spec"]
        pairs = [(v, n1[v]) for v in p["argF2"]]
        mx = max(max(a, b) for a, b in pairs)
        print(f"| {{5..{y}}} | {F} | {F2} | {spec[F]} | {n1[F]} | {F + n1[F]} | "
              f"{F2 - F - n1[F]} | {pairs} | {mx / F:.3f} | {p['D_top']} |")

    print("\n### the top band v >= 0.8 F\n")
    for y in ms:
        p = prof[y]
        F, F2, n1, N, spec = p["F"], p["F2"], p["n1"], p["N"], p["spec"]
        Ntot = p["openings"]
        band = p["top_band"]
        print(f"\nM = {{5..{y}}}  F = {F}  F_2 = {F2}  (F_2 - F = {F2 - F})")
        print("| v | v/F | m(v) | n1(v) | N(v) | Sig=v+n1 | Sig-F | rarity null n1_0 | "
              "n1 - n1_0 |")
        print("|---|---|---|---|---|---|---|---|---|")
        for v in band:
            n0 = rarity_null(spec, Ntot, 2 * spec[v])
            print(f"| {v} | {v / F:.3f} | {spec[v]} | {n1[v]} | {N[v]} | {v + n1[v]} | "
                  f"{v + n1[v] - F:+d} | {n0} | {n1[v] - n0:+d} |")

    print("\n### isolation at the top of the spectrum\n")
    print("| M | top 8 sizes | iso_k | m of top 5 | n1 of top 5 |")
    print("|---|---|---|---|---|")
    for y in ms:
        p = prof[y]
        tops = sorted(p["spec"], reverse=True)[:8]
        iso = [tops[i] - tops[i + 1] for i in range(len(tops) - 1)]
        print(f"| {{5..{y}}} | {tops} | {iso} | {[p['spec'][v] for v in tops[:5]]} | "
              f"{[p['n1'][v] for v in tops[:5]]} |")

    print("\n### exceptionless checks\n")
    tot = bad = 0
    for y in ms:
        p = prof[y]
        for v in p["n1"]:
            tot += 1
            if p["n1"][v] > p["N"][v] - 1:
                bad += 1
    print(f"n1(v) <= N(v) - 1 : {tot - bad} of {tot} realised sizes, {bad} exceptions")
    for y in ms:
        p = prof[y]
        assert max(v + p["n1"][v] for v in p["n1"]) == p["F2"]
    print(f"max_v (v + n1(v)) = F_2 : {len(ms)} of {len(ms)} machines, 0 exceptions")
    print("D_top = " + ", ".join(f"{prof[y]['D_top']}" for y in ms))
    print("larger member of the F_2 pair, / F = "
          + ", ".join(f"{max(max(v, prof[y]['n1'][v]) for v in prof[y]['argF2']) / prof[y]['F']:.3f}"
                      for y in ms))
    # monotonicity of n1 on the top band
    for y in ms:
        p = prof[y]
        band = p["top_band"]
        ups = [(band[i], band[i + 1], p["n1"][band[i + 1]] - p["n1"][band[i]])
               for i in range(len(band) - 1) if p["n1"][band[i + 1]] > p["n1"][band[i]]]
        print(f"  {{5..{y}}} rises of n1 on the top band: {ups}")
    # Sig over the whole spectrum
    print("\nSig(v) = v + n1(v) over the whole spectrum:")
    for y in ms:
        p = prof[y]
        S = {v: v + p["n1"][v] for v in p["n1"]}
        lo = min(S.values())
        hi = max(S.values())
        n_ge = sum(1 for v in S if S[v] >= p["F"])
        print(f"  {{5..{y}}}: F={p['F']} range [{lo}, {hi}], "
              f"{n_ge}/{len(S)} sizes with Sig >= F, "
              f"mean {sum(S.values()) / len(S):.1f}")


if __name__ == "__main__":
    main()
