# Sample run: two closed tree items re-read on the fields construction (2026-09-12)

Method: the fields-explorer skill (.claude/skills/fields-explorer). Scripts
research/stack/r8/layers.py and research/stack/r8/flank_killers.py. The owner's request: pick
one or two items closed as "a count, not a location", "just CRT" or "no pattern" and see whether
the construction gives a lead. A sample, not a lane.

## Item 1. R4.d.i.a, the witness after a square (closed: blind classes + ordinary density, a count)

Field reading. Between consecutive prime squares g^2 and g'^2 the gear rows that can strike are
the gears up to g. The rows of the gears below g are already running; the only NEW row is g,
whose field `higher:g` enters at g^2 (its square). A column (n, n+2) in the layer is open to the
old rows iff no gear below g divides n or n+2. The new row takes, among those old-open columns,
exactly the ones where g divides a member; that strike is g m with m a survivor of the gears
below g in [g, g'^2 / g], a short list with two to four numbers. Every other old-open column of
the layer is a twin (a number below g'^2 with no factor up to g is prime). So, per layer,

    twins in (g^2, g'^2) = old-open columns - toll of the new row,   toll <= #survivors m in [g, g'^2/g].

Measured (layers.py, 166 layers, g from 5 to 997; the assertion "every old-open column not in the
toll is a twin" checked at every layer):

| g -> g' | layer columns | old-open | toll | twins | survivors m (toll bound) | the new row's kills |
|---|---|---|---|---|---|---|
| 5 -> 7 | 3 | 3 | 1 | 2 | 2 | 35 |
| 7 -> 11 | 11 | 6 | 2 | 4 | 4 | 77, 91 |
| 11 -> 13 | 7 | 2 | 0 | 2 | 2 | |
| 13 -> 17 | 19 | 8 | 1 | 7 | 3 | 221 |
| 17 -> 19 | 11 | 2 | 0 | 2 | 2 | |
| 19 -> 23 | 27 | 5 | 1 | 4 | 2 | 437 |
| 23 -> 29 | 51 | 8 | 0 | 8 | 3 | |
| 29 -> 31 | 19 | 2 | 0 | 2 | 2 | |
| 31 -> 37 | 67 | 11 | 0 | 11 | 4 | |
| 37 -> 41 | 51 | 7 | 0 | 7 | 3 | |
| 41 -> 43 | 27 | 3 | 0 | 3 | 2 | |
| 43 -> 47 | 59 | 11 | 0 | 11 | 2 | |

Over the 166 layers: toll share of the old-open columns mean 0.015, maximum 1/3 (at 5 -> 7);
toll zero in 118 layers; toll never reaches its bound; twins per layer minimum 2, median 41;
old-open share of the layer's columns mean 0.076.

What this changes. The finer statement (8e, twins between consecutive prime squares) reads on
the construction as "the wheel of the gears below g leaves more open columns in (g^2, g'^2) than
the new row's toll", and the toll is located (g m, m listed) and almost always zero. The count
that remains is the old wheel's open columns in the layer, a periodic object read in a stretch
placed at the square; the square origin's exact facts (blind classes, parabolic bands, caustics;
programme parts M, Q) are statements about exactly this placement. Step 8 (the section form)
needs the same for the section's first layer only, since a twin in (p^2, p'^2) is in the section
[p^2, P^2) with P the first prime at or above p^2.

Lead, honestly labelled. The reduction moves the unknown from "twins in the layer" to "old-open
columns in the layer", and the latter is the classic short-interval sieve count; the new part is
only that the new row's contribution is located and tiny (toll bound = survivors of a stretch of
length (g'^2 - g^2)/g about 2 gap + gap^2/g). No closed form for the old-open count was found
here; the lead is the object to work on next in the construction: the old wheel's open columns
between g^2 and g'^2 as offsets from g^2, by row (which gear closes each column of the layer, from
which origin), against the caustic law of part Q.

## Item 2. Location rules (closed: nine candidate rules all at chance by p = 13)

Field reading. Who strikes the flanks of a twin: for each column k, the row of the smallest gear
striking column k-1 and column k+1. If twins sat at a preferred flank pattern beyond what the
wheel imposes, the pair frequencies at twins would exceed the frequencies at the wheel's open
columns.

Measured (flank_killers.py, numbers 1000 .. 201000, 33,334 columns, 2,129 twins; the wheel
prediction over a full period of the gears 5 .. 23):

| flank pair (left row, right row) | share at twins | share at all columns | wheel share at open columns |
|---|---|---|---|
| (5, 5) | 0.327 | 0.200 | 0.333 |
| (7, 5) | 0.138 | 0.057 | 0.133 |
| (5, 7) | 0.135 | 0.057 | 0.133 |
| (11, 5) | 0.046 | 0.026 | 0.044 |
| (5, 0) | 0.022 | 0.022 | |

The twins' flank pattern is the wheel's open-column pattern to three decimals. Closed again in
one line: the flanks of a twin are struck by the rows of 5 and 7 at exactly the rate the wheel
gives any open column; no location beyond the wheel.

## Verdict of the sample

One item re-opened as a located object (item 1: the new row's toll between squares is located
and nearly always zero; what remains is the old wheel's open columns at the square origin, the
object to build next, by row and offset). One item closed again (item 2: the wheel). The
construction did what the owner said it would: it split a count into a located part and a
periodic part, and it showed at once when a candidate is only the wheel.
