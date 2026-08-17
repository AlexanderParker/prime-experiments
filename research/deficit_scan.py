"""k-frame deficit D(L) = L - sum_{delta<=L} psi_k(delta) to 10^6.

psi_k(delta) = C * prod_{q|delta, q>=5} (q-2)/(q-4)
                 * prod_{q | 9 delta^2 - 1, q>=5} (q-3)/(q-4),
C = prod_{q>=5} (1 - 4/(q-2)^2) = 0.396880...

This is the k-frame image of the handover's section 5.6 inequality (divide the
adjacent-frame statement by 3). Result: global minimum D = 0.5448 at L = 2,
and D grows roughly like 0.7-0.8 log L (2.27 at 10^2, 4.72 at 10^3, 8.04 at
10^5, 10.87 at 10^6). D(L) >= 0 for all L is the leading-order form of
kappa(L) >= 0; note it controls only L << 1/d (see docs/review-2026-08-17.md,
section 5).
"""
import numpy as np
from math import log

def primes_upto(n):
    s = bytearray([1])*(n+1); s[0:2] = b'\x00\x00'
    for i in range(2, int(n**0.5)+1):
        if s[i]: s[i*i::i] = bytearray(len(s[i*i::i]))
    return [i for i in range(n+1) if s[i]]

def run(Lmax=10**6):
    C = 1.0
    for q in primes_upto(10**7):
        if q >= 5: C *= 1 - 4/(q-2)**2
    print(f"C = {C:.9f}")
    top = 3*Lmax + 2
    spf = np.zeros(top+1, dtype=np.int64)
    for p in primes_upto(int(top**0.5)):
        m = spf[p*p::p]; m[m == 0] = p; spf[p*p::p] = m
    def distinct(n):
        fs = []
        while n > 1:
            p = int(spf[n]) or n
            fs.append(p)
            while n % p == 0: n //= p
        return fs
    S = 0.0
    mn, amn = 1e18, 0
    marks = {10, 100, 1000, 10**4, 10**5, 10**6}
    for delta in range(1, Lmax+1):
        val = C
        for q in distinct(delta):
            if q >= 5: val *= (q-2)/(q-4)
        seen = set()
        for nn in (3*delta-1, 3*delta+1):
            for q in distinct(nn):
                if q >= 5 and q not in seen:
                    seen.add(q); val *= (q-3)/(q-4)
        S += val
        D = delta - S
        if D < mn: mn, amn = D, delta
        if delta in marks:
            print(f"L={delta:8d}  D={D:9.4f}  D/logL={D/log(delta):7.4f}")
    print(f"global min D = {mn:.4f} at L = {amn}")

if __name__ == "__main__":
    run()
