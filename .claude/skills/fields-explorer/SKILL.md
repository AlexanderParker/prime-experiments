---
name: fields-explorer
description: Use when doing pattern work on the primes machine with the owner's fields construction (docs/fields_view.html and research/tools/fields_twin.py) - building or reading fields, turning a visual observation into a tested rule, re-examining tree items that were closed as "a count", "just CRT" or "no pattern"
---

# The fields construction

The owner's reference construction for pattern searching (2026-09-12: "close to what I have in
my imagination"). Every pattern search on the machine is done on it, not on the 6k +- 1 fold.

## The construction, in one paragraph

A field is a grid. Columns are the natural numbers n0 .. n0+nn-1, every integer, never folded
onto the slots 6k +- 1. Rows are gears (primes), one row for every gear that takes part in the
field's strike; a cell is painted where that gear takes part at that number. The sieve is only
a highlight on top: a strike at a number n = 5 mod 6 kills a left member (yellow), at n = 1
mod 6 a right member (red), at an even n or 3 | n it kills nothing (white); a strike at a prime
is a hit on a twin member (green) or on a single prime (blue). A column no row strikes is open:
dark green if n is a twin member, dark blue if a single prime, black if neither. A field is
never a count, never an overlay, never a least-factor summary; those hide which gear struck at
what interval, which is the thing being looked for.

The fields (ids as the twin uses them):

- `multiples`: row g at every multiple of g.
- `squares`: row g at g^2 (the square field; the cuts of the stack).
- `products:j`: numbers with exactly j prime factors with multiplicity, on the row of each
  factor; j runs from 2 to one past the largest count that still kills in range.
- `higher:g`: composites whose smallest gear is g (rows g and the primes above it, up to and
  including the first prime with no kill in range; each row painted where its gear divides n,
  the other residues near-black). This is the gear field of g: g times the survivors of the
  gears below g.
- `higher1:g`: the same with g dividing exactly once.
- `lower:g`: composites whose largest gear is g (rows g and the gears below).
- `lower1:g`: the same with g dividing exactly once.
- an "all" row per field (the column's strike code if any row strikes it, else its open code)
  and a Summary field made of every field's "all" row, reorderable.

The machine: a prime size q, with markers at q, q^2 and q# (the primorial), the mirror point
q#/2 + offset, and cycles of length q# (cycle c starts at baseN0 + (c-1) q#). The machine's
gears are the primes up to q; its window is (q, q^2]; the gears above q are the next machines.

## The tools

- `docs/fields_view.html` is the view (artifact 5f8913e4-d329-41a2-9a2c-9d76e65d47c8). Keys:
  arrows page and change size, Shift+arrows cycle and gear count, +/- double or halve the
  numbers, M mirror, 0 or Esc clear. Marker labels sit below each field; click to highlight a
  range; the dashed line under the top gear highlights the machine's gears; legend entries dim
  a colour. Do not add analysis to the view (no overlays, no counts, no left/right splits as
  fields); the owner looks at it raw.
- `research/tools/fields_twin.py` is the machine-readable twin, same fields and codes:
  `T = Twin(n0, nn, ngears, q, cycle)`; `T.field(name)` gives `.rows`, `.M` (rows x nn int8,
  codes 0-6), `.open` (0 or 7/8/9), `.all`; `T.fields()`, `T.markers()`, `T.to_csv(dir)` (one
  CSV per field plus summary.csv); probes `probe_period(name)` (rows that repeat with q#),
  `probe_mirror(name)` (rows mirror-symmetric about q#/2), `summary(names)`,
  `probe_signature(names)` (per number, which fields strike it), `probe_lone_killers(names)`
  (twin-slot composites killed by exactly one field). CLI:
  `uv run python research/tools/fields_twin.py --n0 1 --nn 400 --gears 11 --q 7 --cycle 1 --probe`.
- Gate any new field or probe against the view on a few cells before using it (25 is a right
  kill on row 5, 35 a left kill, 5 a twin hit, 49 a residue of gear 5's field, 101/103 open twin).

## The method: from a picture to a rule

1. Say what is being looked at in field words: which field, which rows, which range, which
   machine and cycle. A twin is a column no row of any field strikes, so a twin's location is
   the complement of the strikes; "where twins are not" is what the fields show directly.
2. State the observation as a rule about rows and offsets: gear h strikes at offsets c + j h
   from an origin (a square, a cut, a cycle start, a mirror point). Offsets are numbers, not
   residues; give the origin.
3. Test the rule exhaustively over ranges, machine sizes and cycles with the twin, and report
   the failures by number (the first failing column, the gear that struck it). A rule that holds
   by the wheel's own rate is the wheel (compare against a full period of the gears involved);
   say so in one line and stop.
4. Keep a rule only when it locates something the wheel does not: a strike or an opening at a
   fixed numerical offset from an origin, across sizes.
5. Write the result into the fields programme (research/proof/fields_construction.md), the
   proof document (research/proof/proof_skeleton.md, Part IV.3a/b) if it touches step 8, and
   the tree node under R4.d.i.i; scripts in research/stack/r8/.

## Re-examining closed items

Tree items closed as "a count, not a location", "just CRT" or "no pattern" were measured on the
slot fold with overlays. Re-read each on the construction: which rows strike inside the item's
range, from which origin, and whether the item's count is a sum of located strikes. The reading
that turned out useful first (research/proof/fields_sample_run.md): between consecutive squares
g^2 and g'^2 the only new field is `higher:g`, whose strikes there are g m for m a survivor of
the gears below g in [g, g'^2/g], two to four numbers, located; every other column open to the
gears below g is a twin. Keep the honesty rule: if the re-reading reproduces the wheel or a
known count, say so and close it again in one line.

## Words

Engine, gears, column, strike, open, section, cut, stack by squares, machine, window, cycle,
mirror. Never window/rung/ladder/descent in valve or stack work. CRT and the sieve are imprints
of the mechanism, not the mechanism: describe rows and offsets, then name the theorem in a
prior-art line.
