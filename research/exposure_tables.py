"""Tables of prime exposure windows and their relationships to each other and to slots 1, 5.

No stepping, no gcd, no reformulation. These are the raw relationship tables: each gear's
exposure window, how each gear's window sits against every other gear's, and how the windows
line up over the 1 and 5 slots of the 6-cycle. The purpose is to see whether the step count
to the next full alignment is readable from the relationships themselves.

Definitions used throughout, all in twin-slot coordinates `m`, where the pair is
`(6m - 1, 6m + 1)`:

    threat residues of gear q      m = +/- u_q mod q, with u_q = 6^{-1} mod q
    which member each threatens    the "5" member 6m - 1, or the "1" member 6m + 1
    exposure windows               the two arcs between the threats: q - 2u - 1 and 2u - 1
    slip of q against q'           q mod q' - how far q's threat pattern advances per turn
                                   of q measured in q's frame
    forward distance f_q(m)        steps from m to this gear's next threat
"""

import itertools
import sys
from math import prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slip_algebra import gears_upto, tooth


def threat_detail(q):
    """Gear q's two threat residues, and which of the 1 / 5 slots each one attacks."""
    u = tooth(q)
    out = []
    for t in (u, q - u):
        # which member does q divide at m = t?  6t - 1 or 6t + 1
        member = "5-slot (6m-1)" if (6 * t - 1) % q == 0 else "1-slot (6m+1)"
        out.append((t, member))
    return out


def forward_distance(q, m):
    """Steps from slot m to gear q's next threat."""
    u = tooth(q)
    return min((u - m) % q, ((q - u) - m) % q)


def table_a(gears):
    """Each gear's exposure window and which slot its threats attack."""
    rows = []
    for q in gears:
        u = tooth(q)
        det = threat_detail(q)
        rows.append({
            "q": q, "u": u,
            "threat_lo": det[0][0], "attacks_lo": det[0][1],
            "threat_hi": det[1][0], "attacks_hi": det[1][1],
            "long_run": q - 2 * u - 1, "short_run": 2 * u - 1,
            "q_mod_6": q % 6,
        })
    return rows


def table_b(gears):
    """Pairwise: slip both ways, and how the two threat patterns interleave."""
    rows = []
    for q, r in itertools.combinations(gears, 2):
        uq, ur = tooth(q), tooth(r)
        # positions in the combined period where each threatens
        P = q * r
        tq = sorted({m for m in range(P) if m % q in (uq, q - uq)})
        tr = sorted({m for m in range(P) if m % r in (ur, r - ur)})
        both = sorted(set(tq) & set(tr))
        # nearest approach of the two threat sets, and how many coincide
        rows.append({
            "pair": (q, r), "slip_qr": q % r, "slip_rq": r % q,
            "threats_q": len(tq), "threats_r": len(tr),
            "coincident": len(both),
            "first_coincidence": both[0] if both else None,
            "period": P,
        })
    return rows


def table_c(gears):
    """Pairwise joint exposure: how the two windows overlap."""
    rows = []
    for q, r in itertools.combinations(gears, 2):
        uq, ur = tooth(q), tooth(r)
        P = q * r
        exposed = [m for m in range(P)
                   if m % q not in (uq, q - uq) and m % r not in (ur, r - ur)]
        runs = []
        if exposed:
            start = exposed[0]
            prev = exposed[0]
            for m in exposed[1:]:
                if m == prev + 1:
                    prev = m
                    continue
                runs.append(prev - start + 1)
                start = prev = m
            runs.append(prev - start + 1)
            if exposed[0] == 0 and exposed[-1] == P - 1 and len(runs) > 1:
                runs[0] += runs.pop()
        rows.append({
            "pair": (q, r), "period": P, "exposed": len(exposed),
            "expected": (q - 2) * (r - 2),
            "runs": len(runs), "longest": max(runs), "shortest": min(runs),
        })
    return rows


def table_d(m, gears):
    """Forward distance to each gear's next threat, from slot m."""
    return sorted(((forward_distance(q, m), q) for q in gears))


def table_e(m, gears, span):
    """For each step J, every gear that threatens slot m + J."""
    cover = {J: [] for J in range(1, span + 1)}
    for q in gears:
        f = forward_distance(q, m)
        J = f if f > 0 else q
        while J <= span:
            cover[J].append(q)
            J += q
        # the other threat of the same gear
        u = tooth(q)
        for t in (u, q - u):
            J2 = (t - m) % q
            if J2 == 0:
                J2 = q
            while J2 <= span:
                if q not in cover[J2]:
                    cover[J2].append(q)
                J2 += q
    return cover
