"""Why the real teeth help the glue: for how many gears is a small v a LETTER?
Real separations are a_g = (g -+ 1)/3 exactly, so a_g runs 2,2,4,4,6,6,8,10,10 over 5..31 and
every even v <= 10 equals a_g for some gear.  Random symmetric members have a_g = 2 v_g mod g
spread over the whole range.  Counted here: |Leg(v)| = #{g : v = +-2u_g mod g} for v = 6,7,8,
real against 2000 random members."""
import random, sys
from gl_glue import gears_of, us_of

def leg(gears, us, v):
    n = 0
    for g, u in zip(gears, us):
        d = (2 * u) % g
        if v % g and v % g in (d, (-d) % g):
            n += 1
    return n

rng = random.Random(7)
for top in (17, 19, 23, 29, 31):
    gears = gears_of(top)
    real = us_of(gears)
    for v in (6, 7, 8, 10):
        r = leg(gears, real, v)
        cnt = {}
        for _ in range(2000):
            vs = [rng.randrange(1, (g + 1) // 2) for g in gears]
            us = [pow(g - 0, 1, g) for g in gears]  # placeholder, replaced below
            k = 0
            for g, x in zip(gears, vs):
                d = (2 * x) % g
                if v % g and v % g in (d, (-d) % g):
                    k += 1
            cnt[k] = cnt.get(k, 0) + 1
        mean = sum(k * c for k, c in cnt.items()) / 2000
        atleast = sum(c for k, c in cnt.items() if k >= r) / 2000
        print(f"m{top} v={v}: real |Leg(v)|={r}; family mean {mean:.2f}, "
              f"P(|Leg| >= real) = {atleast:.3f}, dist {dict(sorted(cnt.items()))}")
