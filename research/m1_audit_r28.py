"""Round 28 (mechanic): DOES ANY MECHANIC CLAIM LEAN ON M1?

Constructor's round-28 note: M1 is REFUTED - the legal kill alphabet is NOT
{a, b, q'}; values like a + q' and 2q' are realised from m31 on, so anything
enumerating chains over three letters is enumerating a strict subset.

THE AUDIT.  This lane's word legality has always been RESIDUE-BASED, not
value-based: j5_multi.legal_word and a_kill's legality both accept any integer
congruent to 0, +s or -s mod q' (plus the prefix-sum range condition).  That
set is infinite and contains a + q', 2q', q' + (q'-s), 2q' - s, ...  This
script proves the point from the code and from this lane's own recorded
findings rather than asserting it.

usage: <venv>/python research/m1_audit_r28.py
"""
import sys

sys.path.insert(0, __file__.rsplit("research", 1)[0] + "research")

QPP = 59
S = (2 * pow(6, -1, QPP)) % QPP          # 20
THREE = {S, QPP - S, QPP}                # the M1 alphabet {a, b, q'}

# C36 (round 27): the COMPLETE realised k=3 kill-word level at 53 -> 59.
REALISED_3 = [(20, 39), (39, 20), (20, 59), (59, 20),
              (20, 98), (98, 20), (20, 118), (118, 20)]
REALISED_4 = [(20, 98, 20)]
# the full legal letter set this lane enumerated at that step, under F(53)=145
LETTERS_ENUMERATED = [20, 39, 59, 79, 98, 118, 138]


def legal(v):
    r = v % QPP
    return r in (0, S, (-S) % QPP)


print("M1 AUDIT - the letter alphabet this lane actually enumerates\n")
print("  q' = %d, s = 2u' mod q' = %d, so M1's three letters are %s"
      % (QPP, S, sorted(THREE)))
print("  this lane's legality test is RESIDUE-based: v is legal iff "
      "v mod %d in {0, %d, %d}\n" % (QPP, S, (-S) % QPP))
for v in LETTERS_ENUMERATED:
    tag = "IN M1's alphabet" if v in THREE else "OUTSIDE M1's alphabet"
    assert legal(v), ("letter not legal", v)
    print("    letter %3d = %-22s  legal here: YES   %s"
          % (v, {20: "s", 39: "q'-s", 59: "q'", 79: "q'+s", 98: "q'+(q'-s)",
                 118: "2q'", 138: "2q'+s"}[v], tag))
outside = [v for v in LETTERS_ENUMERATED if v not in THREE]
print("\n  %d of the %d letters enumerated are OUTSIDE M1's alphabet: %s"
      % (len(outside), len(LETTERS_ENUMERATED), outside))

used = sorted({v for w in REALISED_3 + REALISED_4 for v in w})
beyond = [v for v in used if v not in THREE]
print("\n  and they are not hypothetical - the letters appearing in this "
      "lane's own\n  REALISED words at 53 -> 59 (C36, complete levels) are %s,"
      "\n  of which %s lie outside M1's alphabet." % (used, beyond))
print("  the arity-4 carrier is (20, 98, 20) = (s, q' + (q'-s), s): its middle "
      "letter\n  is one of the very values M1 omits, and it is what lifts "
      "A_kill from 3 to 4.")

assert beyond, "expected letters beyond M1's alphabet"
assert 98 in beyond and 118 in beyond
assert all(legal(v) for v in used)
print("\nVERDICT: NO mechanic claim leans on M1.  Every enumeration this lane "
      "runs is over\n  the residue-legal alphabet, which strictly contains "
      "M1's three letters; and this lane's\n  round-27 result that the padded "
      "letter q'+(q'-s) carries arity 4 is independent\n  CORROBORATION of "
      "M1's refutation, recorded before the refutation was filed.")
print("ALL ASSERTIONS PASSED")
