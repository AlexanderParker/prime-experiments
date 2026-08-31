/-
INCREMENT-WIDTH CASE-SPLIT CERTIFICATES, step 23->29: the gears as
blocking predicates in (phase, offset) coordinates.  Gear q has teeth
at the slot residues u and q - u with 6u = q -+ 1 (GENERATED).
-/
import CaseSplit

namespace IncCert29

def gb5 (r i : ℕ) : Bool := ((r + i) % 5 == 1) || ((r + i) % 5 == 4)
def gb7 (r i : ℕ) : Bool := ((r + i) % 7 == 6) || ((r + i) % 7 == 1)
def gb11 (r i : ℕ) : Bool := ((r + i) % 11 == 2) || ((r + i) % 11 == 9)
def gb13 (r i : ℕ) : Bool := ((r + i) % 13 == 11) || ((r + i) % 13 == 2)
def gb17 (r i : ℕ) : Bool := ((r + i) % 17 == 3) || ((r + i) % 17 == 14)
def gb19 (r i : ℕ) : Bool := ((r + i) % 19 == 16) || ((r + i) % 19 == 3)
def gb23 (r i : ℕ) : Bool := ((r + i) % 23 == 4) || ((r + i) % 23 == 19)
def gb29 (r i : ℕ) : Bool := ((r + i) % 29 == 24) || ((r + i) % 29 == 5)

end IncCert29
