"""Round 69 / loop entry 74: generate Lucas (Pratt) primality certificates as Lean proofs.

`norm_num` proves a prime by trial division, which costs the square root: a thirteen-digit prime
took six and a half minutes here and still hit the recursion limit, so the kernel certificate of
round 65 stopped at its fourth link.  A Lucas certificate costs the logarithm instead - one
square-and-multiply chain plus one check per prime factor of p - 1 - and every step is a single
multiplication of numbers the size of p, which `norm_num` does instantly.

This emits, for each requested prime, a Lean theorem `Pratt.prime_<p>` proved by
`lucas_primality`, recursing into the prime factors of p - 1 whenever they are too large for
`norm_num` themselves.

usage: uv run python research/stack/r8/gen_pratt.py <prime> [<prime> ...] > proofs/File.lean
"""

import io
import os
import sys

from sympy import factorint, isprime, primerange

SMALL = 10 ** 8  # below this, norm_num proves primality quickly enough


def witness(p, factors):
    """Smallest a whose order modulo p is p - 1."""
    for a in range(2, 200):
        if pow(a, p - 1, p) != 1:
            continue
        if all(pow(a, (p - 1) // q, p) != 1 for q in factors):
            return a
    raise RuntimeError("no witness below 200 for %d" % p)


def chain(a, e, p, tag):
    """Lean lines computing a^e % p by square-and-multiply, ending in hypothesis `tag`."""
    bits = bin(e)[2:]
    lines = []
    exp = 1
    r = a % p
    lines.append("    have %s0 : %d ^ 1 %% %d = %d := by norm_num" % (tag, a, p, r))
    for i, b in enumerate(bits[1:], start=1):
        prev_exp, prev_r = exp, r
        if b == "0":
            exp = 2 * prev_exp
            r = (prev_r * prev_r) % p
            lines.append(
                "    have %s%d : %d ^ %d %% %d = %d :="
                % (tag, i, a, exp, p, r))
            lines.append(
                "      Pratt.sq_of %s%d (by norm_num) (by norm_num)" % (tag, i - 1))
        else:
            exp = 2 * prev_exp + 1
            r = (prev_r * prev_r % p * a) % p
            lines.append(
                "    have %s%d : %d ^ %d %% %d = %d :="
                % (tag, i, a, exp, p, r))
            lines.append(
                "      Pratt.sq_mul_of %s%d (by norm_num) (by norm_num)" % (tag, i - 1))
    return lines, "%s%d" % (tag, len(bits) - 1), r


def prime_term(q, emitted):
    if q in emitted:
        return "prime_%d" % q
    return "(by norm_num : Nat.Prime %d)" % q


def emit(p, emitted, out):
    """Emit a certificate for p, recursing into the factors of p - 1 first."""
    if p in emitted or p < SMALL:
        return
    factors = sorted(factorint(p - 1))
    for q in factors:
        if q >= SMALL:
            emit(q, emitted, out)
    a = witness(p, factors)

    out.append("/-- `%d` is prime, by a Lucas certificate with witness %d. -/" % (p, a))
    out.append("theorem prime_%d : Nat.Prime %d := by" % (p, p))
    out.append("  refine lucas_primality %d ((%d : ℕ) : ZMod %d) ?_ ?_" % (p, a, p))
    out.append("  · rw [Pratt.cast_pow_eq_one_iff (by norm_num)]")
    lines, last, r = chain(a, p - 1, p, "h")
    assert r == 1, "witness chain did not end at 1"
    out.extend(lines)
    out.append("    rw [show (%d : ℕ) - 1 = %d by norm_num, %s]" % (p, p - 1, last))
    out.append("  · intro q hq hqd")
    fac = sorted(factorint(p - 1).items())
    prod = "%d ^ %d" % fac[-1]
    for q, e in reversed(fac[:-1]):
        prod = "%d ^ %d * (%s)" % (q, e, prod)
    out.append("    have hfac : %d - 1 = %s := by norm_num" % (p, prod))
    out.append("    rw [hfac] at hqd")
    n = len(factors)
    for i, q in enumerate(factors):
        last = (i == n - 1)
        if not last:
            out.append("    rcases (Nat.Prime.dvd_mul hq).mp hqd with hq%d | hqd" % i)
            out.append("    . have hqe : q = %d :=" % q)
            out.append("        (Nat.prime_dvd_prime_iff_eq hq %s).mp (hq.dvd_of_dvd_pow hq%d)"
                       % (prime_term(q, emitted), i))
            pre = "      "
        else:
            out.append("    have hqe : q = %d :=" % q)
            out.append("      (Nat.prime_dvd_prime_iff_eq hq %s).mp (hq.dvd_of_dvd_pow hqd)"
                       % prime_term(q, emitted))
            pre = "    "
        out.append("%ssubst hqe" % pre)
        e = (p - 1) // q
        sub, sublast, subr = chain(a, e, p, "g%d" % i)
        for ln in sub:
            out.append(pre + ln[4:])
        out.append("%srw [show (%d : ℕ) - 1 = %d by norm_num, show %d / %d = %d by norm_num]"
                   % (pre, p, p - 1, p - 1, q, e))
        out.append("%sexact Pratt.ne_one_of_mod (by norm_num) %s (by norm_num)" % (pre, sublast))
    out.append("")
    emitted.add(p)


def main(argv):
    ps = [int(x) for x in argv]
    for p in ps:
        if not isprime(p):
            raise SystemExit("%d is not prime" % p)
    out = [
        "/-",
        "PrattCertificates (round 69, 2026-09-17): Lucas certificates for the large primes the",
        "chain certificate needs.  Generated by research/stack/r8/gen_pratt.py - each theorem gives a",
        "witness of order p - 1 and the square-and-multiply chain that checks it.",
        "-/",
        "import PrattTools",
        "",
        "namespace Pratt",
        "",
    ]
    emitted = set()
    for p in ps:
        emit(p, emitted, out)
    out.append("end Pratt")
    text = "\n".join(out) + "\n"
    target = os.environ.get("PRATT_OUT", "-")
    if target == "-":
        io.open(sys.stdout.fileno(), "w", encoding="utf-8", closefd=False).write(text)
    else:
        io.open(target, "w", encoding="utf-8", newline="\n").write(text)
        sys.stderr.write("wrote " + target + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
