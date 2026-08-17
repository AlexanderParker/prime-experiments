"""Slip-path methods: alignment and openings by walking the machine's own state.

No CRT, no modular inverses, no products walked. Everything is gear state
(phase, slip, turn count) advanced by addition.

1. The discovery-moment law. At the moment prime r is discovered, the machine
   knows the slip s = r mod q and the completed turns t = (r - s)/q of the
   lower gear. The next joint block of q and r is then

       s*(q + r) + t*q*(q - s) - s^2

   exactly (algebraically equal to q*r, but computed from state, not from
   multiplying the primes). Tested: every prime pair below 200, zero failures.

2. The path walker. From ANY position n, the next joint reset of q and r:
   jump to r's next reset, then step whole turns of r; each turn advances q's
   phase by the slip; stop when q's phase is 0. At most q turns.

3. The flipped walker (twin-relevant). In slot space (slot k = pair
   (6k-1, 6k+1)), gear q blocks k = +-u_q mod q, u_q = 6^{-1} mod q, and is
   open elsewhere. Walk from slot n to the next slot open to EVERY gear in a
   set: advance one slot at a time, each gear's phase ticking by +1 (the
   diagonal); a blocked slot names its blocker and the walk moves on. Inside
   the certification window (6k+1 <= y^2) the slots this walk lands on are
   exactly the twin pairs - this is the repo's constructor (section 21)
   rebuilt at path level from the slip frame.
"""

def joint_block_from_discovery(q, r):
    s = r % q
    t = r // q
    return s * (q + r) + t * q * (q - s) - s * s

def next_joint_reset(q, r, n):
    """Next m > n with q and r both at reset (m divisible by both)."""
    b = n % r
    pos = n + (r - b if b else r)
    phase = pos % q
    s = r % q
    steps = 1
    while phase:
        pos += r
        phase = (phase + s) % q
        steps += 1
    return pos, steps

def next_joint_opening(gears, n):
    """Next slot k > n open to every gear: k mod q not in {u_q, -u_q}."""
    us = {q: pow(6, -1, q) for q in gears}
    kills = {q: (us[q], (-us[q]) % q) for q in gears}
    k = n
    while True:
        k += 1
        blocker = next((q for q in gears if k % q in kills[q]), None)
        if blocker is None:
            return k

def stride_anatomy(gears, k_from, k_to):
    """Level-1 condition: name every interior slot's blocker in a stride."""
    us = {q: pow(6, -1, q) for q in gears}
    kills = {q: (us[q], (-us[q]) % q) for q in gears}
    return {k: [q for q in gears if k % q in kills[q]] for k in range(k_from + 1, k_to)}

def chain_prediction(opens, gaps, q, max_k=6):
    """Level-2 condition: max new stride after adding gear q, from the old gap
    word alone. In the k-frame the new gear's teeth are separated by
    s = 3^{-1} mod q (NOT adjacent - see the frame-trap note in
    chain_census.py), so a merge of gaps i..i+m qualifies iff the interior
    opening positions all lie in {phi, phi+s} mod q for some phi."""
    s = (2 * pow(6, -1, q)) % q
    best = max(a + b for a, b in zip(gaps, gaps[1:]))   # k=1 always allowed
    n = len(gaps)
    for i in range(n - 1):
        total = gaps[i]
        for m in range(1, max_k):
            if i + m >= n: break
            pos = [opens[i + j] % q for j in range(1, m + 1)]
            total += gaps[i + m]
            st = set(pos)
            ok = len(st) == 1 or (len(st) == 2 and
                 (max(st) - min(st)) % q in (s, (q - s) % q))
            if ok:
                best = max(best, total)
    return best

if __name__ == "__main__":
    # 1. discovery-moment law
    ps = [p for p in range(2, 200) if all(p % d for d in range(2, int(p**0.5) + 1))]
    bad = sum(1 for i, q in enumerate(ps) for r in ps[i+1:]
              if joint_block_from_discovery(q, r) != q * r)
    print(f"discovery law: {sum(range(len(ps)))} prime pairs < 200, failures {bad}")
    for q, r in [(3, 5), (7, 11)]:
        print(f"  {q},{r}: {joint_block_from_discovery(q, r)}")

    # 2. path walker vs brute force
    import random
    random.seed(2)
    ok = 0
    for _ in range(50):
        q, r = sorted(random.sample(ps[1:20], 2))
        n = random.randrange(10, 3000)
        pos, steps = next_joint_reset(q, r, n)
        brute = next(m for m in range(n + 1, n + q * r + 1) if m % q == 0 and m % r == 0)
        ok += pos == brute
    print(f"path walker: 50 random cases, correct {ok}")

    # 3. flipped walker: the openings path for gears up to 13
    gears = [5, 7, 11, 13]
    walk, k = [], 0
    while k < 28:                     # certification window of y = 13: 6k+1 <= 169
        k = next_joint_opening(gears, k)
        if k < 28:
            walk.append(k)
    print(f"open-slot path, gears {gears}, window k < 28: {walk}")
    print("pairs:", [(6*k - 1, 6*k + 1) for k in walk])
    # every one should be a genuine twin pair (certified inside the window)
    def isp(m):
        return m > 1 and all(m % d for d in range(2, int(m**0.5) + 1))
    assert all(isp(6*k - 1) and isp(6*k + 1) for k in walk)
    print("all certified twin pairs: True")
