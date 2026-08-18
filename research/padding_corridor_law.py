"""Round 15 lateral: the general corridor law for padded-pair shapes.

Every opening lies in E = the 15-residue exposed set mod 35 (avoid teeth of
gears 5 and 7). Two padded links of sizes a*q' and b*q' put three consecutive
openings at r, r+a*g, r+(a+b)*g mod 35 with g = q' mod 35. Feasibility is a
pure function of g - so of q' mod 35 alone.
"""
E = sorted(k for k in range(35) if k % 5 not in (1, 4) and k % 7 not in (1, 6))
Eset = set(E)

def feas(g, a, b):
    return [r for r in E if (r + a * g) % 35 in Eset
            and (r + (a + b) * g) % 35 in Eset]

print("shape feasibility by g = q' mod 35 (units = multiples of q'):")
print(f"  {'g':>3} {'(1,1)':>6} {'(1,2)':>6} {'(2,1)':>6} {'(2,2)':>6}   note")
rows = {}
for g in range(35):
    r11, r12, r21, r22 = (len(feas(g, 1, 1)), len(feas(g, 1, 2)),
                          len(feas(g, 2, 1)), len(feas(g, 2, 2)))
    rows[g] = (r11, r12, r21, r22)
    if g % 5 == 0 or g % 7 == 0:
        note = "g shares a factor with 35"
    else:
        note = ""
    print(f"  {g:>3} {r11:>6} {r12:>6} {r21:>6} {r22:>6}   {note}")
print()
print("primes q' of interest, by residue:")
for qp in (23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97):
    g = qp % 35
    r11 = rows[g][0]
    print(f"  q'={qp:>3} (g={g:>2}): adjacent equal padded links "
          f"{'POSSIBLE (' + str(r11) + ' phases)' if r11 else 'IMPOSSIBLE'}")
n_imp = sum(1 for g in range(35) if rows[g][0] == 0 and g % 5 and g % 7)
n_tot = sum(1 for g in range(35) if g % 5 and g % 7)
print(f"\nover the {n_tot} invertible classes mod 35, adjacent equal padded")
print(f"links are corridor-IMPOSSIBLE in {n_imp} of them "
      f"({100*n_imp/n_tot:.0f}%) - it is a coin-flip property of q' mod 35,")
print("not a trend, so the padding ceiling switches on and off with q'.")
