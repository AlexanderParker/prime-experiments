# Family C (caustic) anchors

Anchors: for each ordered pair of distinct gears (g, h), the columns
g*g - 2 + 6i for 1 <= i < e(g,h), where e(g,h) is the first i >= 1 at which
h strikes; K = {h}; deduplicated by column with K sets merged.  Base anchors
(home a = -1, K = all gears; gear pairs a = g with g, g+2 gears, K = all gears
but g and g+2) are merged in by column for the 'C + base' figures.

| q | anchors | verification failures | gears fully covered | first uncovered gear | twins certified (C alone) | twins certified (C + base) | total window twins | struck columns certified | walk length min/median/max | anchors twins / not twins | mean |K(a)| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | 78 | 0 | 3/9 | 5 (0/3) | 0 | 30 | 30 | 0 | 3 / 5 / 8 | 17 / 61 | 3.40 |
| 101 | 839 | 0 | 18/24 | 5 (0/3) | 0 | 201 | 201 | 0 | 3 / 12 / 16 | 103 / 736 | 7.21 |
| 211 | 3655 | 0 | 39/45 | 5 (0/3) | 0 | 626 | 626 | 0 | 3 / 21 / 26 | 361 / 3294 | 12.50 |
| 401 | 12566 | 0 | 71/77 | 5 (0/3) | 0 | 1789 | 1789 | 0 | 3 / 33 / 40 | 904 / 11662 | 20.46 |
| 1009 | 73759 | 0 | 161/167 | 5 (0/3) | 0 | 8278 | 8278 | 0 | 3 / 68 / 79 | 3913 / 69846 | 41.67 |

## Side check (q = 101 only)

pairs (g, h) with h < g: 276

first 10 pairs, in order of g ascending then h ascending, as (g, h, r, e(g,h)):

  (7, 5, 2, 1), (11, 5, 1, 1), (11, 7, 4, 2), (13, 5, 3, 1), (13, 7, 6, 1), (13, 11, 2, 3), (17, 5, 2, 1), (17, 7, 3, 2), (17, 11, 6, 5), (17, 13, 4, 2)

the smallest s >= 1 with s*h - r*r > 0 and (s*h - r*r) = 0 mod 6, same ten pairs:

  (g=7, h=5) s = 2, (g=11, h=5) s = 5, (g=11, h=7) s = 4, (g=13, h=5) s = 3, (g=13, h=7) s = 6, (g=13, h=11) s = 2, (g=17, h=5) s = 2, (g=17, h=7) s = 3, (g=17, h=11) s = 6, (g=17, h=13) s = 4
