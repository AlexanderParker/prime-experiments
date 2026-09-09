"""Full-period real-phase records of the composite machines {7..q} on the anchor's 30-clock.

Machine {7..q} = the primes 7 <= p <= q striking their multiples; anchor 2, 3, 5 the clock.  Period
P = prod of the gears, in cycles of 30.  A slot (30j + e, 30j + e + 2), e in {11, 17, 29}, is struck
iff a gear divides one of its two members.  Record R = the longest cyclic run of consecutive
struck slots over the whole period (in slots), and the same run measured in numbers (the largest
cyclic gap between the lower members of consecutive open slots), which must reproduce the
real twin-candidate gap of research/harvest/r1/jacobsthal_check.md (30, 42, 66, 108, 150, 204
numbers for {5..7}, {5..11}, ..., {5..23}: with 5 in the anchor these are the same machines).
Also the record on the prefix that is the section of the chain whose lower gears these are:
[9, 121) for {7} (base 3, link 1), [25, 841) for {7..23} (base 5, link 1).

Usage: uv run python research/stack/r2/period_record.py         (q = 7 .. 23; about 100 MB)
"""
import os, json, math, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
E = (11, 13, 17, 19, 29, 31)


def longest_cyclic_run(struck):
    """longest run of True in a cyclic boolean array; returns (length, start index)."""
    n = len(struck)
    if struck.all():
        return n, 0
    # rotate so that position 0 is open
    k = int(np.flatnonzero(~struck)[0])
    s = np.roll(struck, -k)
    padded = np.concatenate(([False], s, [False])).astype(np.int8)
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    lens = ends - starts
    if len(lens) == 0:
        return 0, None
    j = int(lens.argmax())
    return int(lens[j]), int((starts[j] + k) % n)


def main():
    out = []
    gears_all = [7, 11, 13, 17, 19, 23]
    t0 = time.time()
    for i in range(len(gears_all)):
        gears = gears_all[:i + 1]
        P = 1
        for g in gears:
            P *= g
        J = P                                            # cycles in the period
        s = np.zeros((6, J), dtype=bool)
        for g in gears:
            inv = pow(30, -1, g)
            for r, e in enumerate(E):
                s[r, (-e * inv) % g::g] = True
        slot = np.stack((s[0] | s[1], s[2] | s[3], s[4] | s[5]))    # (3, J)
        flat = slot.T.reshape(-1)                                    # slot index 3j + i
        R, at = longest_cyclic_run(flat)
        # in numbers: the gap between consecutive open slots' lower members, cyclic
        opens = np.flatnonzero(~flat)
        lower = 30 * (opens // 3) + np.array((11, 17, 29))[opens % 3]
        gaps = np.diff(np.concatenate((lower, [lower[0] + 30 * J])))
        gmax = int(gaps.max())
        gat = int(lower[int(gaps.argmax())])
        rec = {"gears": gears, "period_cycles": int(J), "slots": int(3 * J), "open_slots": int(len(opens)),
               "record_slots": R, "record_start_slot": at,
               "record_start_lower_member": int(30 * (at // 3) + (11, 17, 29)[at % 3]) if at is not None else None,
               "max_gap_numbers": gmax, "max_gap_from": gat}
        # the section prefix, when this machine is the lower machine of a chain's first link
        for lo, hi, base in ((9, 121, 3), (25, 841, 5)):
            if gears == ([7] if base == 3 else [7, 11, 13, 17, 19, 23]):
                # slots with lower member in [lo, hi)
                lower_all = 30 * (np.arange(3 * J) // 3) + np.array((11, 17, 29))[np.arange(3 * J) % 3]
                m = (lower_all >= lo) & (lower_all < hi)
                sec = flat[m]
                padded = np.concatenate(([False], sec, [False])).astype(np.int8)
                d = np.diff(padded)
                lens = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
                rec["section"] = [lo, hi]
                rec["section_slots"] = int(m.sum())
                rec["section_record_slots"] = int(lens.max()) if len(lens) else 0
                rec["section_open_slots"] = int((~sec).sum())
                rec["section_numbers"] = hi - lo
        out.append(rec)
        print(rec, f"({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(RES, "period_record.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
