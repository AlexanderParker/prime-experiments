"""The law table, part 2: the walk, the transforms and the record laws (documents 1 and 3)
on gear sets whose smallest gear is 2, 3, 5, 7, 11, 13.  Exact over full wheel periods."""

import json
import sys
from math import prod

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from common import (  # noqa: E402
    all_struck_counts,
    feasible,
    gap_census,
    mex_progressions,
    mex_simple,
    open_mask,
    open_mask_triple,
    prog_terms,
    record_scan,
    teeth,
    walk_lengths,
)

SETS = [
    [2, 3, 5], [2, 5, 7], [2, 3, 5, 7], [2, 5, 7, 11], [2, 7, 11, 13],
    [2, 3, 5, 7, 11],
    [3, 5, 7], [3, 5, 7, 11], [3, 7, 11, 13], [3, 11, 13, 17],
    [5, 7, 11], [5, 7, 11, 13], [5, 11, 13, 17],
    [7, 11, 13], [7, 11, 13, 17],
    [11, 13, 17], [11, 13, 17, 19],
    [13, 17, 19],
]


def mex_all(gears, W, tooth_fn=teeth):
    return np.array([mex_simple(gears, x, tooth_fn) for x in range(W)])


def main():
    out = []
    for gears in SETS:
        W = prod(gears)
        qp = min(gears)
        m = len(gears)
        mask = open_mask(gears)
        L = walk_lengths(mask)
        rec = {"gears": gears, "W": W, "q'": qp, "m": m}

        # --- document 3 L30: the plain mex form
        M = mex_all(gears, W)
        mism = int((M != L).sum())
        rec["mex_mismatch"] = mism
        rec["mex_hyp_q>2m"] = qp > 2 * m
        # sharpened: mex < q'  ==>  mex = L
        sel = M < qp
        rec["mex_sharp_bad"] = int((M[sel] != L[sel]).sum())
        rec["mex_sharp_n"] = int(sel.sum())
        rec["mex_ge_qp"] = int((~sel).sum())
        rec["mex_bad_all_ge_qp"] = bool(((M != L) & sel).sum() == 0)

        # --- document 3 L32: the general mex over progressions, self-certifying
        F = int(L.max())
        rec["F_scan"] = F
        for B in (2 * m, F, F + 2):
            MB = np.array([mex_progressions(gears, x, B) for x in range(min(W, 200000))])
            LL = L[: len(MB)]
            cert = MB <= B
            rec[f"progmex_B{B}_certified"] = int(cert.sum())
            rec[f"progmex_B{B}_bad"] = int((MB[cert] != LL[cert]).sum())
            rec[f"progmex_B{B}_terms"] = prog_terms(gears, B)

        # --- document 3 L36: the C identities
        C = all_struck_counts(mask, F + 2)
        dist = np.bincount(L, minlength=F + 2)
        rec["C_dist_ok"] = all(int(dist[j]) == C[j] - C[j + 1] for j in range(F + 1))
        cen = gap_census(mask)
        rec["C_census_ok"] = all(
            cen.get(d, 0) == C[d - 1] - 2 * C[d] + C[d + 1] for d in range(1, F + 2)
        )
        rec["C_record_ok"] = F == max(j for j in range(len(C)) if C[j] > 0)
        rec["C_mean_ok"] = int(L.sum()) == sum(C[1:])
        rec["C"] = C[: F + 2]

        # --- document 1 L16/L17/L18: the record as a cover
        try:
            covF = 0
            while feasible(gears, covF + 1):
                covF += 1
            rec["F_cover"] = covF
        except TimeoutError:
            rec["F_cover"] = None
        rec["parity_value"] = 2 * m - (m % 2)
        rec["mult"] = C[F]

        # --- document 1 L14: letters
        rec["letters"] = {g: sorted({2 % g, (g - 2) % g}) for g in gears}

        # --- document 3 L42/L43: the striker-parity bit
        idx = np.arange(W)
        par = np.zeros(W, dtype=np.int64)
        for g in gears:
            r = idx % g
            hit = np.zeros(W, dtype=bool)
            for t in teeth(g):
                hit |= r == t
            par += hit
        even = int((par % 2 == 0).sum())
        odd = W - even
        rec["xor_bias"] = even - odd
        rec["xor_bias_pred"] = prod(g - 2 * len(teeth(g)) for g in gears)
        # longest run of ones in the XOR bit
        x1 = par % 2 == 1
        best = cur = 0
        for v in np.concatenate([x1, x1]):
            cur = cur + 1 if v else 0
            best = max(best, cur)
        rec["xor_run"] = min(best, W)

        # --- document 3 L44: the correlation product
        bad = 0
        for d in range(1, 41):
            meas = int((mask & np.roll(mask, -d)).sum())
            p = 1
            for g in gears:
                forb = {0 % g, (-2) % g, (-d) % g, (-d - 2) % g}
                p *= g - len(forb)
            if meas != p:
                bad += 1
        rec["corr_bad"] = bad
        # the document's closed form c_g(d) in {g-2, g-3, g-4}
        bad2 = 0
        for d in range(1, 41):
            meas = int((mask & np.roll(mask, -d)).sum())
            p = 1
            for g in gears:
                if d % g == 0:
                    p *= g - 2
                elif d % g in (2 % g, (-2) % g):
                    p *= g - 3
                else:
                    p *= g - 4
            if meas != p:
                bad2 += 1
        rec["corr_bad_docform"] = bad2

        # --- document 3 L45: the holes
        rec["holes_pair"] = sorted(d for d in range(1, F + 2) if cen.get(d, 0) == 0)

        # --- document 3 L34/L35: the triple (twin-candidate) view
        tmask = open_mask_triple(gears)
        nt = int(tmask.sum())
        rec["triple_count"] = nt
        rec["triple_pred"] = prod(g - len({0 % g, (-1) % g, (-2) % g}) for g in gears)
        if nt:
            TL = walk_lengths(tmask)
            TM = np.array([mex_simple(gears, x, lambda g: sorted({0 % g, (-1) % g, (-2) % g})) for x in range(W)])
            rec["triple_mex_mismatch"] = int((TM != TL).sum())
            rec["triple_mex_sharp_bad"] = int((TM[TM < qp] != TL[TM < qp]).sum())
            rec["F3"] = int(TL.max())
            rec["F3_pred"] = 3 * m
            tcen = gap_census(tmask)
            rec["holes_triple"] = sorted(d for d in range(1, rec["F3"] + 2) if tcen.get(d, 0) == 0)
        else:
            rec["F3"] = None

        # --- document 3 L40/L41: the spectrum
        if W <= 20000:
            f = np.fft.fft(mask.astype(float)) / W
            a = np.arange(W)
            prodv = np.ones(W, dtype=complex)
            for g in gears:
                ag = (a * pow(W // g, -1, g)) % g
                uu = np.zeros(W, dtype=complex)
                for t in teeth(g):
                    uu -= np.exp(-2j * np.pi * ag * t / g)
                uu = np.where(ag == 0, (g - len(teeth(g))) + 0j, uu)
                prodv *= uu / g
            rec["spec_maxerr"] = float(np.max(np.abs(f - prodv)))
            rec["spec_min_abs"] = float(np.min(np.abs(f)))
        out.append(rec)
        print(json.dumps({k: v for k, v in rec.items()}))
    with open(__file__.rsplit("s2_")[0] + "results/s2_walk.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
