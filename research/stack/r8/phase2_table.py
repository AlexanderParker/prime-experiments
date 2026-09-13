"""Phase 2 table (owner, 2026-09-14): for each machine, the primorial spiral (phase 1) lands at
E; phase 2 tries one flip about the mirror of every gear subset S (M = product of S, k = 1), up
or down, whose landing stays inside the window; hits (twins) listed, misses counted. Rows
ordered by the set's elements. Each block notes the gear count's parity and the top gear's twin
status. Writes research/stack/r8/results_phase2_subsets.txt and docs/phase2_table.html.
Usage: uv run python phase2_table.py nmachines   (machines are the primes from 11 up)
"""
import sys, html
import numpy as np
from pathlib import Path
from sympy import primerange, isprime


def main():
    nm = int(sys.argv[1])
    qs = []
    for p in primerange(11, 10 ** 6):
        qs.append(p)
        if len(qs) >= nm: break
    N = qs[-1] ** 2 + 10
    sieve = np.ones(N + 1, dtype=bool); sieve[:2] = False
    for i in range(2, int(N ** 0.5) + 1):
        if sieve[i]: sieve[i * i::i] = False
    def twin(n): return 0 <= n < N - 2 and bool(sieve[n]) and bool(sieve[n + 2])
    def alt(xs): return sum(g if i % 2 == 0 else -g for i, g in enumerate(xs))
    head = ("Phase 1: the primorial spiral (base = the lower gears with product at most q/2, spiral over the rest descending). "
            "Phase 2: one flip about the mirror of a gear subset S (M = product of S, k = 1), up (+) or down (−) from the landing E; "
            "hit = a twin inside the window (q, q²]. Every subset whose flip stays inside the window is tried. "
            "Rows ordered by the set's elements. Each block notes the gear count (2 through q) odd or even, and whether the top gear q is a "
            "twin member (left: q+2 prime; right: q−2 prime) or solo.")
    blocks = []; plain = [head, ""]
    for q in qs:
        primes = list(primerange(2, q + 1)); base = []; P = 1
        for p in primes:
            if P * p <= q // 2: P *= p; base.append(p)
            else: break
        E = -1 + 2 * P * alt([p for p in primes if p not in base][::-1])
        lim = q * q
        subs = [((), 1)]
        for p in primes:
            subs = subs + [(s + (p,), m * p) for s, m in subs if m * p <= lim]
        hits = []; misses = 0
        for S, M in subs:
            if not S: continue
            for d in (1, -1):
                L = E + 2 * d * M
                if q < L and L + 2 <= q * q:
                    if twin(L): hits.append((S, d, L))
                    else: misses += 1
        hits.sort(key=lambda h: (h[0], -h[1]))
        par = 'even' if len(primes) % 2 == 0 else 'odd'
        top = 'left' if isprime(q + 2) else ('right' if isprime(q - 2) else 'solo')
        status = {'left': 'twin member, left (q, q+2)', 'right': 'twin member, right (q−2, q)', 'solo': 'solo'}[top]
        title = (f"q = {q}: gears {len(primes)} ({par}); top gear {q} {status}; base {base} (P = {P}); landing E = ({E}, {E+2}) "
                 f"twin={twin(E)}; subsets in reach {len(subs)-1}; hits {len(hits)}, misses {misses}")
        plain.append(title); rows = []
        for S, d, L in hits:
            rows.append((f"{{{', '.join(map(str, S))}}} {'+' if d > 0 else '−'}", f"({L}, {L+2})")); plain.append(f"   S = {rows[-1][0]}  ->  {rows[-1][1]}")
        plain.append("")
        trs = "".join(f"<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>" for a, b in rows)
        blocks.append(f'<details open data-par="{par}" data-top="{top}"><summary>{html.escape(title)}</summary><table><tr><th>subset S, direction</th><th>landing</th></tr>{trs}</table></details>')
    Path("research/stack/r8/results_phase2_subsets.txt").write_text("\n".join(plain), encoding="utf-8")
    page = """<title>Phase 2 Table</title>
<style>
:root{--bg:#f6f4ee;--ink:#1c1a17;--mut:#6b665c;--line:#d8d2c4;--pan:#fffdf8}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15140f;--ink:#ece7da;--mut:#a29b8c;--line:#3a372e;--pan:#1e1c16}}
:root[data-theme="dark"]{--bg:#15140f;--ink:#ece7da;--mut:#a29b8c;--line:#3a372e;--pan:#1e1c16}
body{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:0;padding-block:12px;padding-inline:16px}
h1{font-size:18px;margin:0 0 6px;font-weight:600}
p.lead{color:var(--mut);max-width:90ch;margin:0 0 12px}
details{border:1px solid var(--line);background:var(--pan);border-radius:4px;padding:6px 10px;margin:8px 0}
summary{font-family:ui-monospace,monospace;font-size:12.5px;cursor:pointer}
table{border-collapse:collapse;font-family:ui-monospace,monospace;font-size:12.5px;font-variant-numeric:tabular-nums;margin-top:6px}
td,th{padding:1px 10px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:500}
.wrap{overflow-x:auto}
.ctl{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;padding:8px 10px;border:1px solid var(--line);background:var(--pan);border-radius:4px;margin-bottom:8px;position:sticky;top:0;z-index:2;font-size:13px}
button{font:inherit;padding:2px 8px;border:1px solid var(--line);background:var(--pan);color:var(--ink);border-radius:3px;cursor:pointer}
</style>
<h1>Phase 2 table</h1>
<p class="lead">""" + html.escape(head) + """</p>
<div class="ctl"><span>gear count:</span> <label><input type="checkbox" class="fpar" value="odd" checked> odd</label> <label><input type="checkbox" class="fpar" value="even" checked> even</label> &nbsp; <span>top gear:</span> <label><input type="checkbox" class="ftop" value="left" checked> twin member, left</label> <label><input type="checkbox" class="ftop" value="right" checked> twin member, right</label> <label><input type="checkbox" class="ftop" value="solo" checked> solo</label> &nbsp; <button id="collapse">collapse all</button> <button id="expand">expand all</button> <span id="count"></span></div>
<div class="wrap">""" + "".join(blocks) + """</div>
<script>
function applyFilter(){const par=[...document.querySelectorAll('.fpar')].filter(c=>c.checked).map(c=>c.value),top=[...document.querySelectorAll('.ftop')].filter(c=>c.checked).map(c=>c.value);let n=0;document.querySelectorAll('details').forEach(d=>{const show=par.includes(d.dataset.par)&&top.includes(d.dataset.top);d.hidden=!show;if(show)n++;});document.getElementById('count').textContent=n+' machines shown';}
document.querySelectorAll('.fpar,.ftop').forEach(c=>c.addEventListener('change',applyFilter));
document.getElementById('collapse').addEventListener('click',()=>document.querySelectorAll('details').forEach(d=>d.open=false));
document.getElementById('expand').addEventListener('click',()=>document.querySelectorAll('details').forEach(d=>d.open=true));
applyFilter();
</script>"""
    Path("docs/phase2_table.html").write_text(page, encoding="utf-8")
    print(f"machines {qs[0]}..{qs[-1]} ({len(qs)}); page {len(page)} bytes")


if __name__ == "__main__":
    main()
