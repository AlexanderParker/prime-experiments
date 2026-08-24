"""Round 21: independent assert of the two headline witnesses."""
def openings(y_gears, ks):
    out = []
    for k in ks:
        ex = False
        for g in y_gears:
            u = pow(6, -1, g)
            if k % g == u % g or k % g == (-u) % g: ex = True; break
        out.append(not ex)
    return out
g37 = [5,7,11,13,17,19,23,29,31,37]
# F_3(37) = 97 witness: k=990209189833, gaps [37,23,37]
k = 990209189833
pts = [k, k+37, k+60, k+97]
assert all(openings(g37, pts)), "endpoints/interiors must be open"
interior = [k+i for i in range(1, 97) if (k+i) not in pts]
assert not any(openings(g37, interior)), "every non-window slot in span must be blocked"
print(f"F_3(37)=97 witness k={k}: 4 openings at +0,+37,+60,+97, all 94 interior slots blocked - VERIFIED")
# run_3(37;V(41)) witness: k=1120456097388, word (14,41,14) span 69, V=[0,14,27] mod 41
k2 = 1120456097388
pts2 = [k2, k2+14, k2+55, k2+69]
assert all(openings(g37, pts2))
inter2 = [k2+i for i in range(1, 69) if (k2+i) not in pts2]
assert not any(openings(g37, inter2))
gaps = [14, 41, 14]
assert all((g % 41) in (0, 14, 27) for g in gaps)
# padded: the 41-gap's endpoints share a residue mod 41
assert (k2+14) % 41 == (k2+55) % 41
print(f"run_3(37;V(41)) witness k={k2}: word (14,41,14), V-residues ok, padded link shares residue mod 41 - VERIFIED")
