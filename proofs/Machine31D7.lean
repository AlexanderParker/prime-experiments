/-
Machine 31, depth 7: the QUALIFYING 7-WINDOW DICTIONARY - every window of
7 consecutive machine-31 gaps whose 5 interior gaps are all
>= 12 (the floor `2u''` of gear 37).  42 tuples, measured over the
FULL period 33,426,748,355 by `research/qual_dict.py` (which gate-checks its
own output against the corpus ladder at machines 19 and 23).

NOT KERNEL-CHECKED, AND NOT CLAIMED TO BE: that this list CONTAINS every
realised qualifying 7-window is the census hypothesis `E7` of
`Machine31.Census31`.  What IS kernel-checked here is the only thing the rung
consumes from it - that every listed window sums to at most 88.
-/

import Machine31

namespace Machine31

set_option maxRecDepth 1000000
set_option maxHeartbeats 4000000

/-- The realised qualifying 7-windows of machine 31 (census input). -/
def D7 : List (ℕ × ℕ × ℕ × ℕ × ℕ × ℕ × ℕ) :=
  [(2, 13, 17, 12, 13, 12, 1), (2, 12, 18, 12, 13, 12, 1), (1, 14, 18, 12, 13, 12, 1), (3, 14, 18, 12, 13, 12, 1), (2, 15, 16, 12, 15, 12, 1), (3, 12, 13, 12, 18, 12, 1),
  (5, 12, 13, 12, 18, 12, 1), (1, 12, 13, 12, 18, 14, 1), (5, 13, 12, 15, 13, 12, 2), (1, 12, 13, 12, 18, 12, 2), (2, 13, 17, 12, 13, 13, 2), (5, 12, 13, 12, 15, 13, 2),
  (1, 12, 13, 12, 17, 13, 2), (2, 13, 13, 12, 17, 13, 2), (1, 12, 15, 12, 16, 15, 2), (7, 18, 12, 13, 12, 12, 3), (7, 15, 15, 13, 12, 12, 3), (1, 12, 18, 12, 13, 12, 3),
  (3, 13, 12, 15, 13, 12, 3), (10, 13, 15, 12, 15, 12, 3), (3, 12, 13, 15, 12, 13, 3), (1, 12, 13, 12, 18, 14, 3), (5, 15, 12, 16, 12, 12, 5), (2, 13, 15, 12, 13, 12, 5),
  (1, 12, 18, 12, 13, 12, 5), (9, 13, 13, 14, 13, 12, 5), (2, 12, 13, 15, 12, 13, 5), (5, 12, 12, 16, 12, 15, 5), (5, 18, 12, 13, 12, 17, 5), (5, 17, 12, 13, 12, 18, 5),
  (11, 15, 12, 13, 12, 17, 6), (3, 12, 12, 13, 15, 15, 7), (11, 12, 12, 13, 15, 15, 7), (3, 12, 12, 13, 12, 18, 7), (8, 17, 12, 13, 12, 15, 8), (8, 15, 12, 13, 12, 17, 8),
  (11, 15, 12, 13, 12, 17, 8), (5, 12, 13, 14, 13, 13, 9), (3, 12, 15, 12, 15, 13, 10), (7, 15, 15, 13, 12, 12, 11), (6, 17, 12, 13, 12, 15, 11), (8, 17, 12, 13, 12, 15, 11)]

/-- Every listed qualifying 7-window sums to at most `Q_7(31; 12) = 88`. -/
theorem D7_ok : D7.all (fun t => Nat.ble (t.1 + t.2.1 + t.2.2.1 + t.2.2.2.1 + t.2.2.2.2.1 + t.2.2.2.2.2.1 + t.2.2.2.2.2.2) 88) = true := by
  decide +kernel

end Machine31
