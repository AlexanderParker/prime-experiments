/-
Machine 29, depth 7: the QUALIFYING 7-WINDOW DICTIONARY - every window of
7 consecutive machine-29 gaps whose 5 interior gaps are all
>= 10 (the floor `2u''` of gear 31).  46 tuples, measured over the
FULL period 1,078,282,205 by `research/qual_dict.py` (which gate-checks its
own output against the corpus ladder at machines 19 and 23).

NOT KERNEL-CHECKED, AND NOT CLAIMED TO BE: that this list CONTAINS every
realised qualifying 7-window is the census hypothesis `E7` of
`Machine29.Census29`.  What IS kernel-checked here is the only thing the rung
consumes from it - that every listed window sums to at most 71.
-/

import Machine29

namespace Machine29

set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

/-- The realised qualifying 7-windows of machine 29 (census input). -/
def D7 : List (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  [(2, 11, 12, 12, 11, 14, 1), (5, 10, 13, 12, 11, 19, 1), (3, 12, 13, 12, 11, 10, 2), (3, 14, 13, 10, 13, 10, 2), (4, 10, 13, 10, 12, 11, 2), (7, 10, 13, 10, 12, 11, 2),
  (4, 10, 11, 12, 12, 11, 2), (1, 14, 11, 12, 12, 11, 2), (3, 14, 11, 12, 12, 11, 2), (2, 12, 13, 12, 11, 12, 2), (2, 12, 11, 12, 13, 12, 2), (4, 10, 13, 12, 11, 15, 2),
  (3, 12, 13, 12, 11, 15, 2), (3, 12, 13, 12, 11, 17, 2), (3, 12, 11, 12, 20, 10, 3), (3, 12, 11, 10, 22, 10, 3), (3, 15, 14, 10, 11, 12, 3), (3, 17, 15, 10, 11, 12, 3),
  (3, 10, 22, 10, 11, 12, 3), (3, 12, 13, 12, 11, 12, 3), (3, 10, 20, 12, 11, 12, 3), (4, 10, 13, 10, 13, 12, 3), (3, 17, 13, 10, 13, 12, 3), (2, 10, 11, 12, 13, 12, 3),
  (5, 10, 11, 12, 13, 12, 3), (3, 12, 11, 12, 13, 12, 3), (5, 12, 11, 12, 13, 12, 3), (2, 15, 11, 12, 13, 12, 3), (2, 17, 11, 12, 13, 12, 3), (3, 14, 15, 10, 10, 13, 3),
  (2, 11, 12, 12, 11, 14, 3), (2, 10, 13, 10, 13, 14, 3), (5, 10, 11, 12, 13, 14, 3), (3, 13, 10, 10, 15, 14, 3), (3, 12, 11, 10, 14, 15, 3), (3, 12, 13, 10, 13, 17, 3),
  (3, 12, 11, 10, 15, 17, 3), (2, 11, 12, 12, 11, 10, 4), (2, 11, 12, 10, 13, 10, 4), (3, 12, 13, 10, 13, 10, 4), (2, 15, 11, 12, 13, 10, 4), (3, 12, 13, 12, 11, 10, 5),
  (3, 14, 13, 12, 11, 10, 5), (1, 19, 11, 12, 13, 10, 5), (3, 12, 13, 12, 11, 12, 5), (2, 11, 12, 10, 13, 10, 7)]

/-- Every listed qualifying 7-window sums to at most `Q_7(29; 10) = 71`. -/
theorem D7_ok : D7.all (fun t => Nat.ble (t.1 + t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2.1 + t.2.2.2.2.2.1 + t.2.2.2.2.2.2) 71) = true := by
  decide +kernel

end Machine29
