"""Digital twin of the fields explorer (docs/fields_view.html), for a machine to read.

Same construction as the view: columns are the natural numbers n0 .. n0+nn-1; rows are gears
(the first N primes, 2 and 3 included, or a field's own row set); a cell is coded by what the
field's strike at that number is. Codes (integers, so a field is a small int matrix):
    0  nothing (the gear does not take part in the field's strike here)
    1  hit at a twin prime member          (green in the view)
    2  hit at a single prime               (blue)
    3  hit that kills a left member, n = 5 mod 6   (yellow)
    4  hit that kills a right member, n = 1 mod 6  (red)
    5  hit that kills nothing, n even or 3 | n     (white)
    6  a gear's residue that is not a strike of this field (per-gear fields only; near-black)
Open columns (no row of the field strikes n) are reported separately, coded
    7  open, twin prime member   8  open, standalone prime   9  open, neither
The "all" row of a field is the strike code of the column if any row strikes it, else its
open code. Machine: q (a prime), its square, its primorial q#, the cycle c (offset (c-1) q#),
the mirror point q#/2 + offset.

Fields (ids as in the view, in order):
    'multiples'      row g at every multiple of g
    'squares'        row g at g^2
    'products:j'     n with exactly j prime factors (multiplicity), row of each factor; j = 2 .. jmax+1
    'higher:g'       composites whose smallest gear is g, rows g and the primes above it up to the
                     first with no kill in range; each row painted where its gear divides n
    'higher1:g'      the same with g dividing exactly once
    'lower:g'        composites whose largest gear is g, rows g and the gears below it
    'lower1:g'       the same with g dividing exactly once

API (import this module):
    T = Twin(n0, nn, ngears, q, cycle)     -> factorisations, primality, markers computed once
    T.field(name)                          -> Field with .rows (gears), .M (rows x nn int8 matrix), .open (nn int8: 0 or 7/8/9), .all (nn int8)
    T.fields()                             -> the ordered list of field names the view would show
    T.markers()                            -> dict q, q2, qp, mirror, offset, cycle
    T.to_csv(path, names=None)             -> one CSV per field: header row = numbers, one line per gear row, last line = all
    T.probe_period(name)                   -> which rows repeat exactly with the machine's period q# inside the range (a machine-readable version of "the machine's gears are periodic, the others are not")
    T.probe_mirror(name)                   -> which rows are mirror-symmetric about q#/2 within the cycle
    T.summary(names)                       -> (names, matrix fields x numbers of the "all" rows), also written as summary.csv by to_csv
    T.probe_signature(names)               -> per number, the tuple of fields that strike it (what strikes and what does not)
    T.probe_lone_killers(names)            -> twin-slot composites killed by exactly one field of the list, by field
CLI: uv run python fields_twin.py --n0 1 --nn 400 --gears 11 --q 7 --cycle 1 --csv outdir [--field multiples]
"""
import argparse, os
import numpy as np


def _primes_upto(n):
    s = np.ones(n + 1, dtype=bool); s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = False
    return np.nonzero(s)[0]


def _factor(n):
    f = []; m = n; p = 2
    while p * p <= m:
        while m % p == 0: f.append(p); m //= p
        p += 1 if p == 2 else 2
    if m > 1: f.append(m)
    return f


class Field:
    def __init__(self, name, rows, M, open_, all_):
        self.name, self.rows, self.M, self.open, self.all = name, rows, M, open_, all_


class Twin:
    def __init__(self, n0=1, nn=400, ngears=11, q=7, cycle=1):
        self.n0, self.nn, self.ngears, self.q, self.cycle = n0, nn, ngears, q, cycle
        self.n = np.arange(n0, n0 + nn)
        top = max(n0 + nn + 2, q + 2)
        self._isp = np.zeros(top + 1, dtype=bool); self._isp[_primes_upto(top)] = True
        self.fac = [(_factor(int(x)) if x >= 2 else []) for x in self.n]
        self.prime = np.array([len(f) == 1 for f in self.fac])
        self.twin = np.array([self.prime[i] and (self._isp[x - 2] or self._isp[x + 2]) for i, x in enumerate(self.n)])
        gears = []; p = 2
        while len(gears) < ngears:
            if self._isp[p] if p < len(self._isp) else all(p % r for r in range(2, int(p ** 0.5) + 1)): gears.append(p)
            p += 1
        self.gears = gears
        if q not in gears: raise ValueError("q must be one of the gears (a prime at most the largest gear)")
        self.prim = 1
        for g in gears:
            if g <= q: self.prim *= g
        self.offset = (cycle - 1) * self.prim

    # ---- codes
    def _hitcode(self, i):
        if self.twin[i]: return 1
        if self.prime[i]: return 2
        r = int(self.n[i]) % 6
        return 3 if r == 5 else 4 if r == 1 else 5

    def _opencode(self, i):
        if self.n[i] < 2: return 0
        return 7 if self.twin[i] else 8 if self.prime[i] else 9

    # ---- field definitions: mark(h, n, f) and rows
    def _jmax(self):
        jm = 1
        for i, f in enumerate(self.fac):
            x = int(self.n[i])
            if x % 6 in (1, 5) and len(f) >= 2 and len(f) > jm and any(p in self.gears for p in f): jm = len(f)
        return max(jm, 2)

    def _higher_rows(self, g, mark):
        rows = [g]; nmax = int(self.n[-1]); h = g + 1
        while h <= nmax:
            if self._isp[h]:
                rows.append(h)
                kills = any((int(self.n[i]) % 6 in (1, 5)) and mark(h, int(self.n[i]), f) for i, f in enumerate(self.fac))
                if not kills: break
            h += 1
        return rows

    def _def(self, name):
        if name == 'multiples': return self.gears, (lambda h, n, f: n % h == 0), False
        if name == 'squares': return self.gears, (lambda h, n, f: n == h * h), False
        if name.startswith('products:'):
            j = int(name.split(':')[1]); return self.gears, (lambda h, n, f, j=j: len(f) == j and h in f), False
        kind, g = name.split(':'); g = int(g)
        if kind == 'higher':
            mark = lambda h, n, f, g=g: len(f) >= 2 and f[0] == g and h in f
            return self._higher_rows(g, mark), mark, True
        if kind == 'higher1':
            mark = lambda h, n, f, g=g: len(f) >= 2 and f[0] == g and f[1] != g and h in f
            return self._higher_rows(g, mark), mark, True
        if kind == 'lower':
            return [h for h in self.gears if h <= g], (lambda h, n, f, g=g: len(f) >= 2 and f[-1] == g and h in f), True
        if kind == 'lower1':
            return [h for h in self.gears if h <= g], (lambda h, n, f, g=g: len(f) >= 2 and f[-1] == g and f[-2] != g and h in f), True
        raise KeyError(name)

    def fields(self):
        names = ['multiples', 'squares'] + [f'products:{j}' for j in range(2, self._jmax() + 2)]
        for kind in ('higher', 'higher1', 'lower', 'lower1'):
            names += [f'{kind}:{g}' for g in self.gears]
        return names

    def field(self, name):
        rows, mark, residues = self._def(name)
        M = np.zeros((len(rows), self.nn), dtype=np.int8)
        for r, h in enumerate(rows):
            for i, f in enumerate(self.fac):
                x = int(self.n[i])
                if x >= 2 and mark(h, x, f): M[r, i] = self._hitcode(i)
                elif residues and x % h == 0 and x >= 2: M[r, i] = 6
        struck = (M > 0) & (M != 6)
        open_ = np.array([0 if struck[:, i].any() else self._opencode(i) for i in range(self.nn)], dtype=np.int8)
        all_ = np.array([int(M[:, i][struck[:, i]][0]) if struck[:, i].any() else int(open_[i]) for i in range(self.nn)], dtype=np.int8)
        return Field(name, rows, M, open_, all_)

    # ---- the summary: every field's "all" row side by side
    def summary(self, names=None):
        names = names or self.fields()
        S = np.zeros((len(names), self.nn), dtype=np.int8)
        for r, name in enumerate(names): S[r] = self.field(name).all
        return names, S

    def probe_signature(self, names=None):
        """per number: the tuple of fields that strike it (codes 1-5), i.e. what strikes and what does not; returns
        {signature: [numbers]} plus the twins' signatures (which should all be the empty tuple: nothing strikes a twin)"""
        names, S = self.summary(names)
        sig = {}
        for i in range(self.nn):
            key = tuple(names[r] for r in range(len(names)) if 1 <= S[r, i] <= 5 and S[r, i] != 1 and S[r, i] != 2)
            sig.setdefault(key, []).append(int(self.n[i]))
        return names, sig

    def probe_lone_killers(self, names=None):
        """numbers in a twin slot (n = 1 or 5 mod 6, composite) killed by exactly one field of the given list:
        the field that alone accounts for that kill; returns {field: [numbers]}"""
        names, S = self.summary(names)
        out = {}
        for i in range(self.nn):
            x = int(self.n[i])
            if x % 6 not in (1, 5) or self.prime[i] or x < 2: continue
            killers = [names[r] for r in range(len(names)) if S[r, i] in (3, 4)]
            if len(killers) == 1: out.setdefault(killers[0], []).append(x)
        return out

    def markers(self):
        return {'q': self.q + self.offset, 'q2': self.q * self.q + self.offset, 'qp': self.prim + self.offset,
                'mirror': self.prim // 2 + self.offset, 'offset': self.offset, 'cycle': self.cycle, 'period': self.prim}

    def to_csv(self, outdir, names=None):
        os.makedirs(outdir, exist_ok=True)
        sn, S = self.summary(names)
        with open(os.path.join(outdir, 'summary.csv'), 'w') as fh:
            fh.write('field,' + ','.join(str(x) for x in self.n) + '\n')
            for r, name in enumerate(sn): fh.write(f'{name},' + ','.join(str(v) for v in S[r]) + '\n')
        for name in (names or self.fields()):
            F = self.field(name)
            with open(os.path.join(outdir, name.replace(':', '_') + '.csv'), 'w') as fh:
                fh.write('row,' + ','.join(str(x) for x in self.n) + '\n')
                for r, h in enumerate(F.rows): fh.write(f'{h},' + ','.join(str(v) for v in F.M[r]) + '\n')
                fh.write('all,' + ','.join(str(v) for v in F.all) + '\n')

    # ---- probes a machine can run that eyes cannot
    def probe_period(self, name):
        """for each row, does the strike pattern repeat exactly with period q# inside the range (where both copies exist)?"""
        F = self.field(name); P = self.prim; out = {}
        if P >= self.nn: return {h: None for h in F.rows}
        for r, h in enumerate(F.rows):
            a = (F.M[r, :-P] > 0); b = (F.M[r, P:] > 0)
            out[h] = None if not (a.any() or b.any()) else bool(np.array_equal(a, b))
        return out

    def probe_mirror(self, name):
        """for each row, is the strike pattern mirror-symmetric about q#/2 + offset within the cycle in range?"""
        F = self.field(name); c = self.prim // 2 + self.offset; out = {}
        for r, h in enumerate(F.rows):
            ok = True; tested = 0
            for i in range(self.nn):
                x = int(self.n[i]); m = 2 * c - x
                if m < self.n0 or m >= self.n0 + self.nn or m == x: continue
                j = m - self.n0; tested += 1
                if (F.M[r, i] > 0) != (F.M[r, j] > 0): ok = False; break
            out[h] = ok if tested else None
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n0', type=int, default=1); ap.add_argument('--nn', type=int, default=400)
    ap.add_argument('--gears', type=int, default=11); ap.add_argument('--q', type=int, default=7); ap.add_argument('--cycle', type=int, default=1)
    ap.add_argument('--csv', default=None); ap.add_argument('--field', default=None); ap.add_argument('--probe', action='store_true')
    a = ap.parse_args()
    T = Twin(a.n0, a.nn, a.gears, a.q, a.cycle)
    print('gears', T.gears, 'markers', T.markers(), 'fields', len(T.fields()))
    if a.csv: T.to_csv(a.csv, [a.field] if a.field else None); print('csv written to', a.csv)
    if a.probe:
        for name in ([a.field] if a.field else T.fields()[:3]):
            print(name, 'periodic with q#:', T.probe_period(name), 'mirror about q#/2:', T.probe_mirror(name))
        base = [n for n in T.fields() if n in ('multiples', 'squares') or n.startswith('products:')]
        names, sig = T.probe_signature(base)
        print('summary over', base, ': distinct strike signatures', len(sig))
        for key, nums in sorted(sig.items(), key=lambda kv: -len(kv[1]))[:8]: print('  ', key or '(nothing strikes)', len(nums), nums[:12])
        lone = T.probe_lone_killers(base)
        print('twin-slot composites killed by exactly one of these fields:', {k: len(v) for k, v in lone.items()})


if __name__ == '__main__':
    main()
