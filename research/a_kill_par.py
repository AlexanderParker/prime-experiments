"""Round 24 (mechanic): parallel A_kill level driver - the f3_one.py lesson
(independent per-word processes) applied to a_kill.py's word levels.

Enumerates the level's legal words exactly as a_kill.py does (residue +
window + span + overlap prunes), then decides each word in its OWN process
(research/a_kill_word.py, cov_count CRT+SAT, witness assert-verified inside),
POOL at a time.  A word whose process dies abnormally (memory pressure) is
retried, not recorded - crash != ZERO (standing rules 17/21-lesson).

usage: python research/a_kill_par.py y qp kfrom kto [--pool N] [--log F]
"""
import os, subprocess, sys, time
sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")
from a_kill import enumerate_words, legal_values, DEFAULT_CAPS, sub_ok

args = sys.argv[1:]
def popopt(name, default, cast):
    if name in args:
        i = args.index(name); v = cast(args[i+1]); del args[i:i+2]; return v
    return default
POOL = popopt("--pool", 4, int)
LOGP = popopt("--log", None, str)
y, qp = int(args[0]), int(args[1])
kfrom, kto = int(args[2]), int(args[3])
caps = DEFAULT_CAPS[y]
PY = sys.executable
logf = open(LOGP, "a") if LOGP else None

def log(m):
    print(m, flush=True)
    if logf: logf.write(m + "\n"); logf.flush()

s, V, vals = legal_values(y, qp)
log(f"A_kill_par({y} -> {qp}): s={s} V={V}, legal {vals}, caps {caps}, "
    f"pool {POOL}")

# RESUME: prior RESULT lines in the log are deterministic verdicts (realised
# witnesses were assert-verified in their own process; ZERO is a completed
# UNSAT).  Re-reading them avoids re-paying the slow UNSATs after a crash.
known = {}
if LOGP and os.path.exists(LOGP):
    import re
    for l in open(LOGP, encoding="utf-8"):
        m = re.search(r"RESULT m(\d+) word \(([\d, ]+)\) span \d+: (\w+)", l)
        if m and int(m.group(1)) == y:
            w = tuple(int(x) for x in m.group(2).split(","))
            known[w] = 0 if m.group(3) == "ZERO" else 1
    if known:
        log(f"  resumed {len(known)} prior verdicts from {LOGP}")
prev = None
for k in range(kfrom, kto + 1):
    nlet = k - 1
    _, _, _, words = enumerate_words(y, qp, nlet, caps)
    if prev is not None:
        words = [w for w in words if sub_ok(w, prev)]
    log(f"=== k={k}: {len(words)} words to decide ===")
    t0 = time.time()
    res = {w: known[w] for w in words if w in known}
    pending = [w for w in words if w not in known]
    if res:
        log(f"  {len(res)} of {len(words)} known from resume; "
            f"{len(pending)} to decide")
    running = {}       # popen -> word
    while pending or running:
        while pending and len(running) < POOL:
            w = pending[0]
            try:
                p = subprocess.Popen(
                    [PY, os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "a_kill_word.py"), str(y), ",".join(map(str, w))],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True)
            except OSError as e:
                # WinError 1455: commit exhausted - wait, do not drop the word
                log(f"  spawn failed ({e}); waiting 30s")
                time.sleep(30)
                continue
            pending.pop(0)
            running[p] = w
        time.sleep(2)
        for p in [p for p in running if p.poll() is not None]:
            w = running.pop(p)
            out = p.stdout.read()
            line = next((l for l in out.splitlines()
                         if l.startswith("RESULT")), None)
            if p.returncode != 0 or line is None:
                log(f"  RETRY {w}: rc={p.returncode} (no result line - "
                    f"crash, not a verdict)")
                pending.append(w)
                continue
            n = 0 if " ZERO " in line + " " or line.endswith("ZERO") or \
                (": ZERO" in line) else 1
            res[w] = n
            log("  " + line)
    nz = sorted(w for w, n in res.items() if n)
    log(f"N_{k}({y}->{qp}): {len(nz)} realised of {len(res)} decided "
        f"[{time.time()-t0:.0f}s]  realised: {nz}")
    prev = res
    if not nz:
        log(f"==> N_{k} = 0: A_kill({y}->{qp}) = k_max <= {k-1}")
        break
else:
    log(f"==> realised words exist at every level through k={kto}")
