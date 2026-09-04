"""Prover C r33 -- bare-word admissibility at gears 5 and 7 under (T)+(L).
(T) at gear 5 forces v_5 = 1 (the real tooth: (5-1)/2 = 2 is the only other value and it is adjacent);
at gear 7 it allows v_7 in {1, 2} (1 real).  For the PINNED letters (a, b) = (2u', q'-2u') and the
padded letter q', list the legal words (nonzero classes alternating) that are admissible mod 5 with
v_5 = 1 and mod 7 with v_7 = 1 / 2: a word is admissible mod g iff some residue r puts every opening
r, r+w_1, r+w_1+w_2, ... outside the teeth {+-v} mod g (BareAlt.no_gapWord is the necessity half)."""
import itertools, sys
PR = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

def legal_words(a, b, q1, maxlen):
    out = []
    for n in range(1, maxlen + 1):
        for w in itertools.product([0, 1, -1], repeat=n):
            nz = [t for t in w if t]
            if any(nz[i] == nz[i + 1] for i in range(len(nz) - 1)):
                continue
            out.append(tuple(q1 if t == 0 else (a if t == 1 else b) for t in w))
    return out

def admissible(word, g, v):
    teeth = {v % g, (-v) % g}
    for r in range(g):
        x = r
        ok = x not in teeth
        for w in word:
            x = (x + w) % g
            if x in teeth:
                ok = False; break
        if ok:
            return True
    return False

def cls(word, a, b, q1):
    return ''.join('0' if w == q1 else ('a' if w == a else 'b') for w in word)

if __name__ == '__main__':
    maxlen = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print("level q' a b | max word length admissible mod 5 (v=1) / mod 7 (v=1) / mod 7 (v=2) / mod 5&7 (v7=1) / mod 5&7 (v7=2) | literal-only max length, same order | (a,b) admissible mod 5? mod 5&7 v7=1?")
    for i, q1 in enumerate(PR[2:], 2):
        y = PR[i - 1]
        u = round(q1 / 6); a = 2 * u; b = q1 - a
        assert 3 * a in (q1 - 1, q1 + 1)
        words = legal_words(a, b, q1, maxlen)
        def mx(filt):
            ws = [w for w in words if filt(w)]
            return max((len(w) for w in ws), default=0)
        lit = lambda w: q1 not in w
        m5 = mx(lambda w: admissible(w, 5, 1))
        m71 = mx(lambda w: admissible(w, 7, 1)); m72 = mx(lambda w: admissible(w, 7, 2))
        m571 = mx(lambda w: admissible(w, 5, 1) and admissible(w, 7, 1))
        m572 = mx(lambda w: admissible(w, 5, 1) and admissible(w, 7, 2))
        l5 = mx(lambda w: lit(w) and admissible(w, 5, 1))
        l571 = mx(lambda w: lit(w) and admissible(w, 5, 1) and admissible(w, 7, 1))
        l572 = mx(lambda w: lit(w) and admissible(w, 5, 1) and admissible(w, 7, 2))
        ab5 = admissible((a, b), 5, 1); ab57 = ab5 and admissible((a, b), 7, 1)
        aba57 = admissible((a, b, a), 5, 1) and admissible((a, b, a), 7, 1)
        print(f"m{y:<3} q'={q1:<3} a={a:<3} b={b:<3} | {m5} / {m71} / {m72} / {m571} / {m572} | {l5} / {l571} / {l572} | (a,b): {ab5} {ab57}; (a,b,a) mod 5&7 real: {aba57}")
