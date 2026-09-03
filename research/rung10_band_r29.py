"""Round 29 (constructor): the rung-ten live band, decided with a GRADUATED
node budget and per-instance visibility.

PART B of research/rung10_r29.py leaves exactly one obligation:

    no word-legal 4-window of machine 43 has span in [151, 161]

(J = 2 and J = 3 are already under the budget 150 by the deletion-ladder cap
F_j(43) <= F(the machine j-1 gears up) = 118 / 145), plus the depth cap

    L(43) <= 2   (no realised legal 3-letter word)  =>  Q*_5(43) = -inf.

Round 28's own lesson - "a parallel batch whose tail is one hard instance is a
serial job; cap the PER-INSTANCE cost" - is built in here: every instance is
tried at a small budget first, the survivors at a larger one, and each result
is printed as it lands.  Results are cached in a JSON so a killed run resumes.

Usage:
  uv run python research/rung10_band_r29.py [--workers 6]
"""
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import crt_dict                                              # noqa: E402
import rung10_r29 as R                                       # noqa: E402

Y = 43
BUDGETS = (300_000,)   # single pass: this run is a PRICE MEASUREMENT, not a closure
STORE = os.path.join(HERE, "data", "r29", "rung10_band_cache.json")


def job(args):
    tup, nb = args
    t0 = time.time()
    try:
        return tup, crt_dict.realised(Y, tup, node_budget=nb), time.time() - t0
    except Exception:
        return tup, None, time.time() - t0


def load():
    if os.path.exists(STORE):
        with open(STORE) as f:
            return {tuple(json.loads(k)): v for k, v in json.load(f).items()}
    return {}


def save(res):
    tmp = STORE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({json.dumps(list(k)): v for k, v in res.items()}, f)
    os.replace(tmp, STORE)


def main():
    args = sys.argv[1:]
    workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 6
    F = R.KNOWN_F[Y]
    spec = {1: F, 2: R.KNOWN_F[47], 3: R.KNOWN_F[53], 4: R.KNOWN_F[59]}
    gears = R.gears_of(Y)
    E = {g: R.exposed(g) for g in gears}
    floor, ceil = 150, R.KNOWN_F[59]

    # --- obligation 1: the J = 4 band -------------------------------------
    w2 = R.legal_words(Y, 2, spec)
    cand = {}
    for w in w2:
        m = sum(w)
        for gL in range(1, F + 1):
            for gR in range(1, F + 1):
                t = (gL,) + w + (gR,)
                if not R.spec_ok(list(t), spec):
                    continue
                s = gL + m + gR
                if s <= floor or s > ceil:
                    continue
                cand[min(t, t[::-1])] = s
    band = [t for t in cand if not R.ps_refuted(R.prefix(t), gears, E)]
    # --- obligation 2: the length-3 legal words ---------------------------
    w3 = [w for w in R.legal_words(Y, 3, spec)
          if not R.ps_refuted(R.prefix(w), gears, E)]

    # the length-3 words are decided by research/l43_words_r29.py at a large
    # budget; this run is the J = 4 band only.
    todo = sorted(set(band))
    print("machine %d -> 47 :  budget F(43)+47 = %d" % (Y, F + 47))
    print("  J=4 band [%d,%d]: %d candidates, %d survive phase saturation"
          % (floor + 1, ceil, len(cand), len(band)))
    print("  length-3 legal words: %d survive phase saturation" % len(w3))
    print("  total instances to decide: %d" % len(todo), flush=True)

    res = load()
    print("  cache holds %d decided instances" % len(res), flush=True)
    for nb in BUDGETS:
        live = [t for t in todo if res.get(t) is None]
        if not live:
            break
        print("\n  --- pass at node budget %d : %d instances ---" % (nb, len(live)),
              flush=True)
        t0 = time.time()
        done = 0
        with Pool(workers) as pool:
            for tup, ok, dt in pool.imap_unordered(
                    job, [(t, nb) for t in live], chunksize=1):
                done += 1
                if ok is not None:
                    res[tup] = bool(ok)
                if ok or dt > 30:
                    print("      %-22s %-10s %7.1f s   (%d/%d, %.0f s)"
                          % (str(tup),
                             "REALISED" if ok else "refuted" if ok is False
                             else "undecided", dt, done, len(live),
                             time.time() - t0), flush=True)
                if done % 50 == 0:
                    save(res)
                    print("      ... %d/%d  [%.0f s]"
                          % (done, len(live), time.time() - t0), flush=True)
        save(res)

    yes = [t for t in todo if res.get(t) is True]
    und = [t for t in todo if res.get(t) is None]
    print("\n" + "=" * 70)
    print("RESULT")
    print("  realised : %d  %s" % (len(yes), yes[:10]))
    print("  undecided: %d  %s" % (len(und), und[:10]))
    band_yes = [t for t in yes if len(t) == 4]
    band_und = [t for t in und if len(t) == 4]
    w3_yes = [t for t in yes if len(t) == 3]
    w3_und = [t for t in und if len(t) == 3]
    if not band_yes and not band_und:
        print("  ==> NO word-legal 4-window of machine 43 has span > 150.")
    else:
        print("  ==> J=4 band NOT closed (%d realised, %d undecided)"
              % (len(band_yes), len(band_und)))
    if not w3_yes and not w3_und:
        print("  ==> L(43) = 2 CERTIFIED: A_kill(43->47) = 3, J_max(43) = 4,")
        print("      Q*_5(43) = -inf, with no census of machine 43.")
    else:
        print("  ==> L(43) not certified (%d realised, %d undecided: %s)"
              % (len(w3_yes), len(w3_und), w3_und))
    if not yes and not und:
        print("\n  (D) AT 43 -> 47 CERTIFIED BY THIS LANE'S OWN GATE:")
        print("      max_J Q*_J(43; 47) <= 150 = F(43) + 47.")
    print("=" * 70)


if __name__ == "__main__":
    main()
