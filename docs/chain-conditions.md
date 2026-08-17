# The conditions that determine stride, and the chain census

Scripts: `research/slip_path.py` (condition functions), `research/chain_census.py` (census + anatomy).
Session of 2026-08-17, following the openings/slip line. Everything below is exact and deterministic -
no distributions, no trends.

## Level 1: what determines a stride inside a fixed machine

The stride (gap to the next joint opening) from any open slot is the **mex of the union of the gears'
tooth schedules** - the walk ends at the first forward offset no tooth reaches. The certificate is
complete and per-slot: e.g. the maximal stride of gears {5,7,11,13} (11 slots, from slot 122) with every
interior slot's blocker:

    123:[11] 124:[5] 125:[7] 126:[5] 127:[7] 128:[13] 129:[5] 130:[11] 131:[5] 132:[7,13]

Supply is capped by the one-kill lemma: gear q contributes at most 2(floor(L/q)+1) teeth to a stretch of
L, and in the maximal stride above, gears 11 and 13 sit exactly at their ceilings. Long strides require
near-maximal efficiency from every gear (the corpus's section 37b, reproduced at slot level).

## Level 2: what determines stride growth when a gear is added

The chain condition (gear-recursion.md section 4a), verified here in the k-frame from an independent
implementation: adding gear q merges k+1 old strides into one exactly when the k interior openings all
lie in **{phi, phi + s} mod q, with s = 3^{-1} mod q** - the two teeth of the new gear at their true
separation. The new maximum stride is then computable from the old gap word alone. Verified exactly:
predictions 18, 25, 34 for F_k(17), F_k(19), F_k(23) against their true values.

**The frame trap, recorded.** A first version used the adjacent window {phi, phi+1}. Its k=2 count came
out exactly prod(q-4) - the domino count - for every q, which exposed the error: dominoes fit an
adjacent window trivially, but the k-frame teeth are never adjacent (s = +-1 mod q would need q | 2 or
q | 4). The handover's warning about confusing the two frames (section 0.5) is earned; the anomaly in
the data caught it.

## The census (correct window)

    machine        q :  17    19    23    29    31    37    41    43
    gears<=13, k=2 :  72    60    20    12    12     0     0     -
    gears<=17, k=2 :   -  1088   494   380   380    64    12     -
    gears<=19, k=2 :   -     - 11488  9452  9492  2836   632   632
    gears<=19, k=3 :   -     -    62     0     4     0     0     0

* Qualifying interior distances are = 0 or +-s (mod q), so the **minimum qualifying distance is
  min(s, q-s) = (q +- 1)/3** - the k-frame deletion-spacing law (the adjacent-frame version, >= q-1,
  proved in gear-recursion.md section 4; divide by 3).
* Chains die entirely once (q +- 1)/3 > F_k(M) - the saturation threshold, visible as the zeros.
* The 62 at (gears<=19, q=23) reproduces the corpus's census number independently.
* Maximum k observed anywhere: 3. New data point: k=3 also occurs at q=31 (4 runs).

## Anatomy of the maximal chains

All 62 k=3 runs at (gears<=19, q=23) have interior distance word **(s, q-s) = (8,15) or its mirror
(15,8)** - residues a -> a+s -> a, span exactly q, 31 of each orientation. Maximal chains are the
minimal alternation, nothing else occurs. A k=4 run would require the exact consecutive gap word
(s, q-s, s), span q+s, or a distance = 0 mod q (a single gap of exactly q) adjacent to the pattern -
enumerable conditions on the gap word, none present at this size.

## The span law, and what it cannot do

Provable by pigeonhole on the two residue values: same-residue openings are >= q apart, and alternating
distance pairs sum to >= q, so a run of k openings has **span >= floor((k-1)/2) * q**. Combined with
"consecutive openings <= F_k(M) apart" this bounds k only when F_k(M) < q/2 - and the regime that
matters has F_k(M) >> q. So gap structure alone cannot bound k (consistent with the corpus's section
5.5); what remains is the arithmetic of which specific gap words occur - the k=4 requirement above is
the concrete first instance: **does the word (s, q-s, s) ever occur in the gap word of a machine, and
how often?** That is a question about consecutive gap values pinned to exact residues of the next gear,
i.e. the joint distribution of adjacent gaps modulo q - the same object pathway 7.1 needs, now reduced
to specific forbidden/required words.

## Status

Established here: the level-1 and level-2 conditions as working code; the k-frame deletion-spacing
constant (q +- 1)/3; three new exact chain predictions verified; the census and its collapse with q;
the complete anatomy of every maximal chain at the sizes reachable; the span law.

Open, sharpened: bound the occurrences of the chain-extending words ((s, q-s, s) and gap = 0 mod q
adjacent to an (s, q-s) alternation) in the gap word of a machine. Each such word is a forced
configuration in the sense of forbidden-configurations.md, so the minimal-size law and the factorisation
law apply to it - that is the natural next tool to bring to bear.
