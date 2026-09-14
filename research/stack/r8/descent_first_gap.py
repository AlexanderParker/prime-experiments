"""(a) The first gap of each primorial's stripes against the smallest machine in its range.

P_s serves the machines q with P_prev <= q/2 < P_s, i.e. 2 P_prev <= q < 2 P_s.  The smallest
such machine q_min is the first prime at least 2 P_prev; its reach on the t-line is
t <= (q_min^2 - 1) / (2 P_s).  The first gap of the stripes (first t with 2 t P_s +- 1 both
prime) must sit inside that reach for the descent to land on a twin at q_min; at larger q in the
range the reach only grows.  Reported per primorial up to 31#.

usage: uv run python research/stack/r8/descent_first_gap.py
"""
from sympy import primerange, nextprime, isprime
from pathlib import Path

def main():
    out = [__doc__.strip(), ""]
    out.append(f"{'P_s':>14} {'base to':>7} {'P_prev':>12} {'q_min':>10} {'reach t<=':>10} {'first gap t':>11} {'gaps in reach (first 8)':<40} {'landing at first gap'}")
    P = 1; prev = 1
    for p in primerange(2, 40):
        P *= p
        if P < 30: prev = P; continue
        qmin = nextprime(2 * prev - 1)
        reach = (qmin * qmin - 1) // (2 * P)
        gaps = []; t = 1
        while len(gaps) < 8 and t <= max(reach, 2000):
            if isprime(2 * t * P - 1) and isprime(2 * t * P + 1): gaps.append(t)
            t += 1
        inreach = [g for g in gaps if g <= reach]
        out.append(f"{P:>14} {p:>7} {prev:>12} {qmin:>10} {reach:>10} {gaps[0] if gaps else '-':>11} {str(inreach):<40} {(2 * gaps[0] * P - 1, 2 * gaps[0] * P + 1) if gaps else '-'}")
        prev = P
    Path("research/stack/r8/results_descent_first_gap.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[2:]))

if __name__ == "__main__":
    main()
