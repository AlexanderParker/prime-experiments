# anchor 2,3,5. middle runs: gear q's hit points are the numbers q*m with q*m in an open anchor slot
# (n mod 30 in {1,11,13,17,19,29}); six per period 30q, at q*m for m in M_q (mod 30).
import numpy as np
OPEN = np.array([1, 11, 13, 17, 19, 29])
def mset(q): return [m for m in range(1, 30) if (q * m) % 30 in set(OPEN.tolist())]
gears = [37, 41, 43, 47, 53, 59, 61, 67, 71]
X = 300000
n = np.arange(X + 1)
isopen = np.isin(n % 30, OPEN)
print("gear: hit points q*m, m = M_q mod 30; first period's six")
hit_by = {}
for q in gears:
    h = isopen & (n % q == 0)
    hit_by[q] = h
    pts = np.flatnonzero(h)
    print(f"  {q}: M = {mset(q)}; hits at {pts[:6].tolist()} then +{30*q} each period; untouched runs between hits (numbers): {np.diff(pts[:7]).tolist()}")
print()
# collisions: points hit by two gears
anyhit = np.zeros(X + 1, dtype=np.int8)
for q in gears: anyhit += hit_by[q]
col = np.flatnonzero(anyhit >= 2)
print(f"collisions (one open number hit by two gears) below {X}: {len(col)}; first: {col[:8].tolist()} = " + ", ".join(
    "*".join(str(q) for q in gears if c % q == 0) + f"*{c // np.prod([q for q in gears if c % q == 0])}" for c in col[:8]))
print()
# joint untouched: open numbers not hit by any of the nine gears
joint = isopen & (anyhit == 0)
pred = isopen.sum() * np.prod([1 - 1 / q for q in gears])
print(f"open numbers <= {X}: {isopen.sum()}; untouched by all nine gears: {joint.sum()} (independent-fraction prediction {pred:.0f})")
# joint untouched runs, measured in open numbers between consecutive hit points of any gear
hp = np.flatnonzero(anyhit > 0)
gaps = np.diff(hp)
print(f"hit points of any gear: {len(hp)}; gaps between consecutive hit points: mean {gaps.mean():.1f}, median {np.median(gaps):.0f}, max {gaps.max()} at [{hp[gaps.argmax()]}, {hp[gaps.argmax()+1]}]")
top = np.argsort(-gaps)[:8]
print("  longest joint untouched runs: " + ", ".join(f"[{hp[i]},{hp[i+1]}] len {gaps[i]}" for i in top))
# where long runs sit: distance of run centre to nearest multiple of 30q for each gear (inside that gear's clean zone?)
print("  for the longest run, which gears' clean end zones contain its centre:")
for i in top[:3]:
    c = (hp[i] + hp[i + 1]) // 2
    inz = [q for q in gears if min(c % (30 * q), 30 * q - c % (30 * q)) < (7 * q if q % 30 in (7, 23) else q)]
    print(f"    run [{hp[i]},{hp[i+1]}] centre {c}: inside clean zones of {inz}")
# gap distribution vs exponential with the same mean
print(f"  gap distribution: fraction of gaps > 2*mean {np.mean(gaps > 2 * gaps.mean()):.3f} (exponential would give {np.exp(-2):.3f}), > 3*mean {np.mean(gaps > 3 * gaps.mean()):.4f} (exp {np.exp(-3):.4f})")
