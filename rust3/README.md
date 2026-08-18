# gearsuite

Prime and twin-prime gap algorithms in the **slot frame**, built only from laws
this project proved. No probabilistic primality, no fitted constants, no trial
division by numbers the laws show cannot divide.

```
cargo build --release --manifest-path rust3/Cargo.toml
rust3/target/release/gapsuite bench 100000000
```

## The frame

Every prime `> 3` is `6k-1` or `6k+1`. Call `k` a **slot**; it carries the pair
`(6k-1, 6k+1)`. A twin pair is a slot whose members are both prime. Working in
slots is a free 6x compression and it is the frame in which the laws are stated.

## Law → code

| law | statement | where |
|---|---|---|
| tooth law | gear `q >= 5` blocks slot `k` iff `k = ±u (mod q)`, `6u = 1 (mod q)` | `slot::teeth` |
| closed-form tooth | `u' = round(q/6)` — no modular inversion needed | `slot::tooth_offset` |
| slot cap | no gear divides both members (it would divide 2) | two independent bitsets in `sieve` |
| reflection | the two teeth sum to `q` | asserted in `slot` tests |
| onset | gear `q` blocks nothing below `q^2` | strike offsets in `sieve::Segment::sieve` |
| horizon | gears `< y` decide the window `(y, y^2)` exactly | `sieve::gears_for` |
| corridor mod 35 | gears 5,7 leave exactly 15 twin-eligible residues, `15 = (5-2)(7-2)` | `corridor`, `sieve::twin_eligible` |
| self-blindness | twin `(6m-1, 6m+1)` blocks slot `6m^2` | `slot::product_slot` |
| merge law | `F(M+q')` computable from the **old** machine alone | `machine::f_next` |
| literal cap | a literal chain has at most 6 members, every gear, forever | `machine::LITERAL_CAP` |
| tooth separation | `3^-1 (mod q)` is the correct k-frame separation | `slot::tooth_separation` |

The corridor is applied as a **wheel**: gears 5 and 7 are never struck in the
inner loop, their pattern is a 35-entry table, and twin search skips 20 of every
35 slots with no work at all.

## What it computes

```
gapsuite next <n>              next prime after n            (exact, horizon-decided)
gapsuite prev <n>              previous prime before n
gapsuite gaps <from> <count>   prime gaps, streaming
gapsuite twins <from> <count>  twin pairs and their gaps in slots
gapsuite maxgap <from> <to>    largest prime gap in a range
gapsuite ladder <y>            record gaps F of machines {5..y}
gapsuite merge <y> <q>         F(M(y) + q) from the old machine alone
gapsuite bench <n>             throughput
```

## Verification

`cargo test --release` — 24 tests, all green. They check the code against
independently known values, not against itself:

- every segment agrees with trial division, at the origin and at offset `10^6`;
- `pi(10^8) = 5761455`;
- twin pairs below `10^6`: **8168** in the frame. The standard count is 8169
  because it includes `(3, 5)` — which has no slot at all, since 3 is divisible
  by 3. The difference is a property of the frame, not a miss;
- the record-gap ladder reproduces the corpus values `F(2,y) = 6, 15, 21, 33,
  54, 75` (adjacent frame = 3x the slot frame);
- the gap spectra reproduce the project's independently computed
  `[11,16,23,26,28,31]` at machine 13 and `[25,31,35,38,47,50]` at machine 19;
- the merge law predicts `F(M+q')` from the old machine and is cross-checked
  against brute-force construction of the new machine at four steps;
- known maximal gap: 132 after 1357201.

## Measured

On one core, release build:

| task | result |
|---|---|
| sieve to `10^8` | 177 ms — 94 M slots/s (≈ 560 M integers/s) |
| `next_prime(10^12)` | 1000000000039 |
| `F(M(13) + 17)` via merge law | 18, in 0.26 ms, without touching the new period |
| `F(M(19) + 23)` via merge law | 34, in 35 ms; the new machine's period is 37 million |

The last row is the point of the merge law: the new machine's period is the old
one times `q'`, so constructing it directly costs a factor `q'` more than
reading the answer off the old machine.
