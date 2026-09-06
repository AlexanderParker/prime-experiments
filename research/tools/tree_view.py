"""Render research/proof/theory_tree.md as an interactive tree page (docs/theory_tree.html).

The markdown stays the source of truth (agents edit it, git diffs it); this script is a view.
Run from the repo root:  uv run python research/tools/tree_view.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research" / "proof" / "theory_tree.md"
OUT = ROOT / "docs" / "theory_tree.html"

VERDICTS = ["KERNEL", "PROVED", "STRONG", "TESTED", "CANDIDATE", "PARTIAL", "OPEN", "WEAK",
            "UNPROVED", "INSTRUMENT", "FACT", "ROOT", "CLOSED", "DONE", "DEAD", "REFUTED"]
GROUP = {"KERNEL": "proved", "PROVED": "proved", "STRONG": "strong", "DONE": "strong", "TESTED": "strong",
         "CANDIDATE": "candidate", "PARTIAL": "partial", "OPEN": "open", "WEAK": "weak",
         "UNPROVED": "weak", "INSTRUMENT": "fact", "FACT": "fact", "CLOSED": "fact",
         "ROOT": "root", "DEAD": "dead", "REFUTED": "dead"}
_TAIL = r"(?=\s*(?:[,.:;()\-\d]|as\b|in\b|with\b|at\b|for\b|by\b|this\b|but\b|out\b|on\b|of\b|$))"
VERDICT_RE = re.compile(r"(?:^|[.;:)\]]\s+|\(\s*)(" + "|".join(VERDICTS) + r")" + _TAIL)
VERDICT_ANY = re.compile(r"\b(" + "|".join(VERDICTS) + r")\b" + _TAIL)
BULLET_RE = re.compile(r"^(\s*)- (.*)$")
ID_RE = re.compile(r"^\**([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)\.\**\s+(.*)$")


def parse():
    text = SRC.read_text(encoding="utf-8")
    lines = text.splitlines()
    heads = {l.strip(): i for i, l in enumerate(lines) if l.startswith("## ")}
    t0 = heads["## The tree"] + 1
    t1 = min(i for h, i in heads.items() if i > t0)
    log0 = heads["## Log"] + 1
    profile = "\n".join(lines[heads["## Project profile (read by the theory-tree skill; everything project-specific lives here)"] + 1:t0 - 1]).strip()

    nodes, stack, cur = [], [], None
    for line in lines[t0:t1]:
        m = BULLET_RE.match(line)
        if m and (not cur or len(m.group(1)) <= cur["indent"] + 2 or True):
            indent = len(m.group(1))
            body = m.group(2).strip()
            if cur is not None and not re.match(r"^\**[A-Za-z0-9.]+\.\**\s", body) and (cur["lines"][-1].rstrip()[-1:] in "-+=(,/" or re.match(r"^[A-Z]_", body)):
                cur["lines"].append("- " + body)  # a wrapped line that happens to start with "- "
                continue
            idm = ID_RE.match(body)
            node = {"id": idm.group(1) if idm else "", "first": (idm.group(2) if idm else body).replace("**", ""),
                    "indent": indent, "lines": [body.replace("**", "")], "children": []}
            while stack and stack[-1]["indent"] >= indent:
                stack.pop()
            (stack[-1]["children"] if stack else nodes).append(node)
            stack.append(node)
            cur = node
        elif cur is not None and line.strip():
            cur["lines"].append(line.strip())

    def finish(n, depth=0):
        full = " ".join(n["lines"])
        n["text"] = full
        head = n["first"]
        cut = re.search(r"\.\s|:\s", head[4:])
        par = re.search(r"\(", head[12:])
        pos = min([c for c in [(cut.start() + 4) if cut else None, (par.start() + 12) if par else None] if c is not None], default=110)
        n["title"] = head[:pos].strip().rstrip(".")
        body = full[len(n["id"]) + 2:] if n["id"] else full
        vm = VERDICT_RE.search(body) or VERDICT_ANY.search(body)
        n["verdict"] = vm.group(1) if vm else "OPEN"
        n["group"] = GROUP[n["verdict"]]
        n["depth"] = depth
        for c in n["children"]:
            finish(c, depth + 1)
        n["desc"] = sum(1 + c["desc"] for c in n["children"])
        del n["lines"], n["first"], n["indent"]
    for n in nodes:
        finish(n)

    log = []
    for line in lines[log0:]:
        m = BULLET_RE.match(line)
        if m:
            log.append(m.group(2).strip())
        elif log and line.strip():
            log[-1] += " " + line.strip()
    return {"profile": profile, "tree": nodes, "log": log}


TEMPLATE = r"""<title>Twin-Prime Theory Tree</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--bg:#f5f3ee;--panel:#fbfaf7;--ink:#1f2328;--muted:#6a6e75;--line:#dcd8cf;--accent:#1f4e79;--accent-ink:#ffffff;
--proved:#2e7d4f;--strong:#3d8f5f;--candidate:#1f4e79;--partial:#7a5c8a;--open:#b7791f;--weak:#9a7b2f;--fact:#4a5a70;--root:#2d3e55;--dead:#a23b2b;
--chip-ink:#ffffff;--sel:#e9eef5;--hover:#f0ede6;--mono:'IBM Plex Mono',ui-monospace,Menlo,Consolas,monospace;--serif:'IBM Plex Serif',Georgia,serif;--sans:'IBM Plex Sans',system-ui,Segoe UI,Helvetica,Arial,sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15181c;--panel:#1c2026;--ink:#e6e3dc;--muted:#9a9ea6;--line:#2c323a;--accent:#7fb0dc;--accent-ink:#0f1a26;
--proved:#5fbf88;--strong:#6cc48f;--candidate:#7fb0dc;--partial:#b493c4;--open:#e0a94a;--weak:#c9a54f;--fact:#8fa0b8;--root:#a9b6c8;--dead:#e07a68;--chip-ink:#0f1216;--sel:#243040;--hover:#232930}}
:root[data-theme="dark"]{--bg:#15181c;--panel:#1c2026;--ink:#e6e3dc;--muted:#9a9ea6;--line:#2c323a;--accent:#7fb0dc;--accent-ink:#0f1a26;
--proved:#5fbf88;--strong:#6cc48f;--candidate:#7fb0dc;--partial:#b493c4;--open:#e0a94a;--weak:#c9a54f;--fact:#8fa0b8;--root:#a9b6c8;--dead:#e07a68;--chip-ink:#0f1216;--sel:#243040;--hover:#232930}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 var(--sans)}
header{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:baseline;padding:14px 18px 10px;border-bottom:1px solid var(--line);background:var(--panel)}
header h1{font:600 20px/1.2 var(--serif);margin:0;text-wrap:balance}
header .sub{color:var(--muted);font-size:13px}
.counts{display:flex;flex-wrap:wrap;gap:6px;margin-left:auto}
.chip{display:inline-block;padding:1px 8px;border-radius:999px;font:500 11px/18px var(--mono);letter-spacing:.03em;color:var(--chip-ink);white-space:nowrap}
.chip.proved{background:var(--proved)}.chip.strong{background:var(--strong)}.chip.candidate{background:var(--candidate)}.chip.partial{background:var(--partial)}
.chip.open{background:var(--open)}.chip.weak{background:var(--weak)}.chip.fact{background:var(--fact)}.chip.root{background:var(--root)}.chip.dead{background:var(--dead)}
.chip.off{opacity:.35}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 18px;border-bottom:1px solid var(--line)}
.toolbar input{flex:1 1 220px;min-width:160px;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);font:13px var(--sans)}
.toolbar button{padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel);color:var(--ink);font:13px var(--sans);cursor:pointer}
.toolbar button.on{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
.toolbar button:focus-visible,.node>.row:focus-visible,.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
main{display:grid;grid-template-columns:minmax(320px,46%) 1fr;min-height:calc(100vh - 110px)}
#treepane{border-right:1px solid var(--line);overflow:auto;padding:10px 8px 40px}
#detail{overflow:auto;padding:16px 22px 60px;max-width:78ch}
#mappane{grid-column:1/-1;display:none;padding:8px;overflow:auto}
#mappane svg{display:block;width:100%;height:auto}
.node{margin:0}
.row{display:flex;align-items:flex-start;gap:6px;padding:3px 6px;border-radius:6px;cursor:pointer}
.row:hover{background:var(--hover)}.row.sel{background:var(--sel)}
.tw{flex:0 0 14px;color:var(--muted);font:12px var(--mono);line-height:20px;text-align:center}
.id{flex:0 0 auto;font:500 12px/20px var(--mono);color:var(--accent);min-width:38px}
.tt{flex:1 1 auto;line-height:20px}
.tt small{color:var(--muted);font:11px var(--mono)}
.kids{margin-left:14px;border-left:1px solid var(--line);padding-left:4px}
.hid{display:none}
.dim{opacity:.38}
#detail h2{font:600 22px/1.2 var(--serif);margin:0 0 6px;text-wrap:balance}
#detail .meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center;color:var(--muted);font:12px var(--mono);margin-bottom:12px}
#detail p{margin:0 0 12px;white-space:pre-wrap;word-wrap:break-word}
#detail .kidlist{margin:14px 0 0;padding:0;list-style:none}
#detail .kidlist li{padding:6px 0;border-top:1px solid var(--line);display:flex;gap:8px;align-items:baseline}
#detail .kidlist a{color:var(--accent);text-decoration:none;cursor:pointer}
#detail .kidlist a:hover{text-decoration:underline}
.log li{padding:6px 0;border-top:1px solid var(--line)}
.legend{color:var(--muted);font-size:12px;padding:6px 18px 0}
@media (max-width:820px){main{grid-template-columns:1fr}#treepane{border-right:0;border-bottom:1px solid var(--line);max-height:48vh}#detail{max-width:none}}
@media (prefers-reduced-motion:no-preference){.row{transition:background .12s}}
</style>
<header>
  <h1>Twin-Prime Theory Tree</h1>
  <span class="sub">every branch, its verdict, and what it left behind. Source: research/proof/theory_tree.md</span>
  <div class="counts" id="counts"></div>
</header>
<div class="toolbar">
  <input id="q" type="search" placeholder="search titles and text (e.g. domino, clutch, d_0, quiet zone)" aria-label="search">
  <button id="bExp">expand all</button><button id="bCol">collapse to depth 2</button>
  <button id="bUnex" title="OPEN, WEAK, UNPROVED, PARTIAL, CANDIDATE: avenues not yet exhausted">unexhausted only</button>
  <button id="bMap">map view</button><button id="bLog">log</button>
</div>
<div class="legend">Verdict chips filter the tree (click to toggle). Nodes nest by descent: a child is the theory made from a pattern seen while testing its parent.</div>
<main>
  <div id="treepane"></div>
  <div id="detail"><p class="sub">Select a node.</p></div>
  <div id="mappane"></div>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
const DATA = __DATA__;
const GROUPS = ["proved","strong","candidate","partial","open","weak","fact","root","dead"];
const LABEL = {proved:"KERNEL / PROVED",strong:"STRONG",candidate:"CANDIDATE",partial:"PARTIAL",open:"OPEN",weak:"WEAK",fact:"FACT",root:"ROOT",dead:"DEAD"};
const UNEX = new Set(["open","weak","partial","candidate"]);
let on = new Set(GROUPS), unexOnly = false, query = "", selected = null, showLog = false, showMap = false;
const all = []; (function walk(ns, parent){ for (const n of ns){ n.parent = parent; all.push(n); walk(n.children, n);} })(DATA.tree, null);
all.forEach((n,i)=>{ n.key = i; n.open = n.depth < 2; });

function counts(){ const c = {}; GROUPS.forEach(g=>c[g]=0); all.forEach(n=>c[n.group]++);
  const el = document.getElementById("counts"); el.innerHTML = "";
  GROUPS.forEach(g=>{ const s = document.createElement("button"); s.className = "chip "+g+(on.has(g)?"":" off"); s.textContent = LABEL[g]+" "+c[g]; s.style.border="0"; s.style.cursor="pointer";
    s.onclick = ()=>{ if(on.has(g)) on.delete(g); else on.add(g); render(); }; el.appendChild(s); }); }

function matches(n){ if(!on.has(n.group)) return false; if(unexOnly && !UNEX.has(n.group)) return false;
  if(query){ const q = query.toLowerCase(); return (n.id+" "+n.title+" "+n.text).toLowerCase().includes(q);} return true; }
function anyDesc(n){ return matches(n) || n.children.some(anyDesc); }

function build(ns, host){ for(const n of ns){ const vis = anyDesc(n); const d = document.createElement("div"); d.className = "node"+(vis?"":" hid");
  const r = document.createElement("div"); r.className = "row"+(selected===n?" sel":"")+(matches(n)?"":" dim"); r.tabIndex = 0;
  const tw = document.createElement("span"); tw.className = "tw"; tw.textContent = n.children.length ? (n.open?"−":"+") : "·";
  const id = document.createElement("span"); id.className = "id"; id.textContent = n.id || "";
  const tt = document.createElement("span"); tt.className = "tt"; tt.innerHTML = esc(n.title)+" <span class='chip "+n.group+"'>"+n.verdict+"</span>"+(n.desc?" <small>"+n.desc+" below</small>":"");
  r.append(tw,id,tt);
  r.onclick = (e)=>{ if(e.target===tw && n.children.length){ n.open=!n.open; render(); return;} selected = n; if(query||unexOnly) {} render(); detail(n); };
  r.onkeydown = (e)=>{ if(e.key==="Enter"){ selected=n; render(); detail(n);} if(e.key==="ArrowRight"){n.open=true;render();} if(e.key==="ArrowLeft"){n.open=false;render();} };
  d.appendChild(r);
  if(n.children.length){ const k = document.createElement("div"); k.className = "kids"+((n.open||query||unexOnly)?"":" hid"); build(n.children, k); d.appendChild(k); }
  host.appendChild(d); } }

function esc(s){ return s.replace(/[&<>]/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

function render(){ counts(); const tp = document.getElementById("treepane"); tp.innerHTML = ""; build(DATA.tree, tp);
  document.getElementById("bUnex").className = unexOnly?"on":""; document.getElementById("bMap").className = showMap?"on":""; document.getElementById("bLog").className = showLog?"on":"";
  document.getElementById("mappane").style.display = showMap?"block":"none";
  document.getElementById("treepane").style.display = showMap?"none":"block"; document.getElementById("detail").style.display = showMap?"none":"block";
  if(showMap) drawMap(); }

function path(n){ const p=[]; for(let x=n; x; x=x.parent) p.unshift(x.id||x.title.slice(0,20)); return p.join(" › "); }
function detail(n){ const d = document.getElementById("detail"); showLog=false;
  let h = "<h2>"+(n.id?n.id+". ":"")+esc(n.title)+"</h2><div class='meta'><span class='chip "+n.group+"'>"+n.verdict+"</span><span>depth "+n.depth+"</span><span>"+n.desc+" descendants</span><span>"+esc(path(n))+"</span></div>";
  h += "<p>"+esc(n.text)+"</p>";
  if(n.children.length){ h += "<h3 style='font:600 15px var(--serif);margin:16px 0 4px'>Children</h3><ul class='kidlist'>"+n.children.map(c=>"<li><span class='chip "+c.group+"'>"+c.verdict+"</span><a data-k='"+c.key+"'>"+(c.id?c.id+". ":"")+esc(c.title)+"</a></li>").join("")+"</ul>"; }
  if(n.parent){ h += "<p style='margin-top:14px'><a data-k='"+n.parent.key+"' style='color:var(--accent);cursor:pointer'>↑ parent: "+(n.parent.id?n.parent.id+". ":"")+esc(n.parent.title)+"</a></p>"; }
  d.innerHTML = h; d.querySelectorAll("a[data-k]").forEach(a=>a.onclick=()=>{ const m = all[+a.dataset.k]; selected=m; for(let x=m.parent;x;x=x.parent) x.open=true; render(); detail(m); d.scrollTop=0; }); }

function logView(){ const d = document.getElementById("detail"); d.innerHTML = "<h2>Log</h2><div class='meta'><span>"+DATA.log.length+" entries, newest last</span></div><ul class='log' style='padding:0;list-style:none'>"+DATA.log.map(l=>"<li>"+esc(l)+"</li>").join("")+"</ul>"; }

function drawMap(){ const pane = document.getElementById("mappane"); pane.innerHTML = "";
  const root = d3.hierarchy({title:"ROOT", children: DATA.tree, group:"root", id:"", verdict:"", depth:-1}, d=>d.children);
  const W = Math.max(900, pane.clientWidth-16), R = W/2 - 40;
  const tree = d3.cluster().size([2*Math.PI, R - 120]);
  tree(root);
  const svg = d3.create("svg").attr("viewBox", [-W/2, -W/2, W, W]).attr("aria-label","radial map of the theory tree");
  const css = getComputedStyle(document.documentElement); const col = g=>css.getPropertyValue("--"+g).trim();
  svg.append("g").attr("fill","none").attr("stroke", col("line")).attr("stroke-width",1.2)
    .selectAll("path").data(root.links()).join("path").attr("d", d3.linkRadial().angle(d=>d.x).radius(d=>d.y));
  const node = svg.append("g").selectAll("g").data(root.descendants()).join("g")
    .attr("transform", d=>`rotate(${d.x*180/Math.PI-90}) translate(${d.y},0)`);
  node.append("circle").attr("r", d=>d.depth===0?7:(d.data.desc>8?5.5:4)).attr("fill", d=>col(d.data.group||"root")).attr("stroke", col("panel")).attr("stroke-width",1)
    .style("cursor","pointer").on("click",(e,d)=>{ if(d.depth===0) return; const m = all[d.data.key]; selected=m; showMap=false; for(let x=m.parent;x;x=x.parent) x.open=true; render(); detail(m); });
  node.append("title").text(d=>(d.data.id?d.data.id+". ":"")+d.data.title+(d.data.verdict?"  ["+d.data.verdict+"]":""));
  node.filter(d=>d.depth<=2 || (d.data.desc||0)>=4 || !d.children).append("text").attr("dy","0.32em")
    .attr("x", d=>d.x < Math.PI === !d.children ? 8 : -8).attr("text-anchor", d=>d.x < Math.PI === !d.children ? "start" : "end")
    .attr("transform", d=>d.x >= Math.PI ? "rotate(180)" : null).attr("font-size", d=>d.depth<=1?13:10).attr("font-family", css.getPropertyValue("--sans")).attr("fill", col("ink"))
    .text(d=>d.depth===0?"ROOT":((d.data.id?d.data.id+" ":"")+d.data.title).slice(0,46));
  pane.appendChild(svg.node()); }

document.getElementById("q").oninput = e=>{ query = e.target.value.trim(); render(); };
document.getElementById("bExp").onclick = ()=>{ all.forEach(n=>n.open=true); render(); };
document.getElementById("bCol").onclick = ()=>{ all.forEach(n=>n.open=n.depth<2); render(); };
document.getElementById("bUnex").onclick = ()=>{ unexOnly=!unexOnly; render(); };
document.getElementById("bMap").onclick = ()=>{ showMap=!showMap; render(); };
document.getElementById("bLog").onclick = ()=>{ showLog=!showLog; showMap=false; render(); if(showLog) logView(); else if(selected) detail(selected); };
try{ const k = localStorage.getItem("tt-sel"); if(k!==null && all[+k]) { selected = all[+k]; for(let x=selected.parent;x;x=x.parent) x.open=true; } }catch(e){}
render(); if(selected) detail(selected); else { const r = all.find(n=>n.depth===0); if(r){ selected=r; render(); detail(r);} }
new MutationObserver(()=>{ try{ if(selected) localStorage.setItem("tt-sel", String(selected.key)); }catch(e){} }).observe(document.getElementById("detail"), {childList:true});
</script>
"""


def main():
    data = parse()
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    OUT.write_text(html, encoding="utf-8")
    n = sum(1 for _ in iter_nodes(data["tree"]))
    print(f"wrote {OUT} ({n} nodes, {len(data['log'])} log entries, {OUT.stat().st_size // 1024} KB)")


def iter_nodes(ns):
    for n in ns:
        yield n
        yield from iter_nodes(n["children"])


if __name__ == "__main__":
    main()
