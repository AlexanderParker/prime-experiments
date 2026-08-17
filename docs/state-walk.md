# State walks: the machine read in umbrella language

Tools: `state_walk`, `mex_jump`, `gear_state` in `research/slip_path.py`. Each slot is annotated
with one letter per gear - `L`/`R` killing tooth (left/right member), `S` shield, `s` short
umbrella, `U` long umbrella - and every kill names its member. `mex_jump` reproduces the stride
with no stepping (mex of the gears' tooth offsets), verified to agree with the walk.

## Small examples

Gears <= 23, from slot 4 (state order 5,7,11,13,17,19,23):

    slot  5 (29,31): SUUUUUU  ALL UMBRELLAS       (gear 5 shielding, all others umbrella)

Gears <= 47, from slot 8:

    slot  9 (53,55): RURUUUUUUUUUU  [5R,11R]      (product kill 55 = 5*11)
    slot 10 (59,61): SUsUUUUUUUUUU  ALL UMBRELLAS

## The maximal stride of the y=19 machine, walked end to end

25 slots from open slot 110 to 135, all 24 blocked slots with named killers (order 5,7,11,13,17,19):

    111 (665,667):  LLsUUL  [5L,7L,19L]     <- entry anchor: triple kill
    112 (671,673):  USLUUs  [11L]
    113 (677,679):  URUUUs  [7R]
    114 (683,685):  RUUUUS  [5R]
    115 (689,691):  SUULUs  [13L]
    116 (695,697):  LUUsRs  [5L,17R]
    117 (701,703):  UUUSsR  [19R]
    118 (707,709):  ULUssU  [7L]
    119 (713,715):  RSRRSU  [5R,11R,13R]    <- mid anchor: deep hub, 715 = 5*11*13
    120 (719,721):  SRsUsU  [7R]
    121 (725,727):  LUSUsU  [5L]
    122 (731,733):  UUsULU  [17L]
    123 (737,739):  UULUUU  [11L]
    124 (743,745):  RUUUUU  [5R]
    125 (749,751):  SLUUUU  [7L]
    126 (755,757):  LSUUUU  [5L]
    127 (761,763):  URUUUU  [7R]
    128 (767,769):  UUULUU  [13L]
    129 (773,775):  RUUsUU  [5R]
    130 (779,781):  SURSUL  [11R,19L]
    131 (785,787):  LUssUs  [5L]
    132 (791,793):  ULSRUs  [7L,13R]
    133 (797,799):  USsURS  [17R]
    134 (803,805):  RRLUss  [5R,7R,11L]     <- exit anchor: triple kill
    135 (809,811):  SUUUss  ALL UMBRELLAS   -> (809,811), twins

## Findings

1. **Maximal strides are bracketed by coincidence hubs.** Entry (111) and exit (134) are triple
   kills, and the mid-run slot 119 = (713,715) is the recurring deep hub 715 = 5*11*13 from the
   triple-lift analysis. Long strides anchor on many-factor slots.
2. **The smallest gear carries its ceiling load.** Gear 5 kills 9 of the 24 slots - exactly its
   2-per-5 rhythm, the extremal-efficiency law (twin-prime-program.md section 37b) visible line by
   line.
3. **Shields appear uselessly all through the run.** Twelve of the 24 blocked slots have some gear
   shielding while another kills - a shield protects its pair from one gear only. The picture of
   why isolated shields never make twins and only full umbrella stacks do.
4. **Most of the stride is single-point failure.** 18 of 24 slots die to exactly one gear: the
   stride is a chain of fragile links held together by two heavy anchors - the same shape the
   chain-condition/fuel analysis found from the gap-word side (docs/chain-conditions.md).
