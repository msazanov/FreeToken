"""Render the full prompt-private benchmark ledger as one offline dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_events(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.exists():
        return []
    return [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _json_for_script(value: Any) -> str:
    """Avoid closing the script tag if an error message contains HTML syntax."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def render_dashboard(ledger: Path, output: Path) -> Path:
    """Write a self-contained dashboard containing every ledger event exactly once."""
    events = _load_events(ledger)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_for_script(events)
    output.write_text(
        f"""<!doctype html>
<meta charset=\"utf-8\">
<title>FreeToken benchmark ledger</title>
<style>
body {{ margin:24px; background:#101317; color:#e8edf2; font:14px/1.4 system-ui,sans-serif }}
h1 {{ margin:0 0 6px; font-size:20px }} h2 {{ margin-top:28px }} h3 {{ margin:0 0 8px; font-size:15px }} p {{ color:#aeb8c2 }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr)); gap:22px }}
svg {{ width:100%; border:1px solid #38434e; background:#151a20 }} .axis text {{ fill:#cbd5df }} .axis path,.axis line,.gridline {{ stroke:#45525d }}
.legend {{ display:flex; flex-wrap:wrap; gap:12px; color:#cbd5df }} .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px }}
table {{ border-collapse:collapse; width:100%; margin-top:22px; font-size:12px }} th,td {{ border-bottom:1px solid #2b343d; padding:6px; text-align:left; vertical-align:top }} th {{ position:sticky; top:0; background:#101317 }} .fail {{ color:#ff8c8c }} code {{ white-space:pre-wrap; color:#b8c5d0 }}
</style>
<h1>FreeToken — полный журнал бенчмарков</h1>
<p id=summary></p>
<h2>Live prefill sweep</h2><p id=sweep-summary></p>
<h3>Recommended next check</h3><p id=sweep-next></p>
<div class=grid><section><h3>Prefill speed</h3><svg id=sweep-prefill viewBox=\"0 0 620 360\"></svg></section><section><h3>Decode speed</h3><svg id=sweep-decode viewBox=\"0 0 620 360\"></svg></section></div>
<h2>Historical context measurements</h2><div class=grid><section><h3>Speed by context</h3><svg id=speed viewBox=\"0 0 620 360\"></svg></section></div>
<h2>Все попытки</h2><table><thead><tr><th>время</th><th>модель</th><th>контекст</th><th>параметры</th><th>prefill</th><th>decode</th><th>статус</th><th>артефакт / ошибка</th></tr></thead><tbody id=rows></tbody></table>
<script src=\"https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js\"></script>
<script>
const events={encoded};
const ok=events.filter(d=>d.status==='success'), bad=events.filter(d=>d.status!=='success');
document.getElementById('summary').textContent=`${{events.length}} attempts: ${{ok.length}} success, ${{bad.length}} failed. Every event is retained; no request or response body is present.`;
const liveSweep=ok.filter(d=>d.model==='Ornith 1.5 35b' && Number.isFinite(d.parameters?.max_prefill_length) && d.completion_tokens>=1000 && Number.isFinite(d.prefill_tps) && Number.isFinite(d.decode_tps)).sort((a,b)=>a.parameters.max_prefill_length-b.parameters.max_prefill_length);
const bestPrefill=d3.greatest(liveSweep,d=>d.prefill_tps), bestDecode=d3.greatest(liveSweep,d=>d.decode_tps);
document.getElementById('sweep-summary').textContent=liveSweep.length ? `Compared at 16K input + forced 4K output. Fastest prefill: p${{bestPrefill.parameters.max_prefill_length}} (${{bestPrefill.prefill_tps.toFixed(2)}} tok/s). Best observed decode: p${{bestDecode.parameters.max_prefill_length}} (${{bestDecode.decode_tps.toFixed(2)}} tok/s).` : 'No completed forced long-decode sweep events yet.';
document.getElementById('sweep-next').textContent=liveSweep.length ? `Repeat p${{bestDecode.parameters.max_prefill_length}} and p1280 under the same cold-start protocol; select by median decode, while p${{bestPrefill.parameters.max_prefill_length}} is the current prefill-only winner.` : 'Run a forced long-decode sweep first.';
const color=d3.scaleOrdinal(d3.schemeTableau10).domain([...new Set(events.map(d=>d.model||'unknown'))]);
function panel(id, draw) {{ const svg=d3.select('#'+id), W=620,H=360,m={{l:65,r:22,t:20,b:55}},w=W-m.l-m.r,h=H-m.t-m.b; svg.selectAll('*').remove(); draw(svg,W,H,m,w,h); }}
function axes(svg,x,y,m,w,h,xlabel,ylabel) {{ svg.append('g').attr('class','axis').attr('transform',`translate(0,${{m.t+h}})`).call(d3.axisBottom(x)); svg.append('g').attr('class','axis').attr('transform',`translate(${{m.l}},0)`).call(d3.axisLeft(y)); svg.append('text').attr('fill','#cbd5df').attr('x',m.l+w/2).attr('y',m.t+h+43).attr('text-anchor','middle').text(xlabel); svg.append('text').attr('fill','#cbd5df').attr('transform',`translate(16,${{m.t+h/2}}) rotate(-90)`).attr('text-anchor','middle').text(ylabel); }}
function sweepPanel(id,metric,label,color) {{ panel(id,(svg,W,H,m,w,h)=>{{ const x=d3.scaleLinear().domain(d3.extent(liveSweep,d=>d.parameters.max_prefill_length)).nice().range([m.l,m.l+w]), extent=d3.extent(liveSweep,d=>d[metric]), pad=Math.max(.25,(extent[1]-extent[0])*.15), y=d3.scaleLinear().domain([extent[0]-pad,extent[1]+pad]).range([m.t+h,m.t]); axes(svg,x,y,m,w,h,'max_prefill_length (p)',label); svg.append('g').selectAll('line').data(y.ticks(5)).join('line').attr('class','gridline').attr('x1',m.l).attr('x2',m.l+w).attr('y1',d=>y(d)).attr('y2',d=>y(d)); svg.append('path').datum(liveSweep).attr('fill','none').attr('stroke',color).attr('stroke-width',2).attr('d',d3.line().x(d=>x(d.parameters.max_prefill_length)).y(d=>y(d[metric]))); svg.append('g').selectAll('circle').data(liveSweep).join('circle').attr('cx',d=>x(d.parameters.max_prefill_length)).attr('cy',d=>y(d[metric])).attr('r',5).attr('fill',color).append('title').text(d=>`p${{d.parameters.max_prefill_length}}: ${{d[metric].toFixed(3)}} tok/s`); }}); }}
sweepPanel('sweep-prefill','prefill_tps','prefill, tok/s','#66c2a5');
sweepPanel('sweep-decode','decode_tps','decode, tok/s','#fc8d62');
panel('speed',(svg,W,H,m,w,h)=>{{ const values=ok.filter(d=>Number.isFinite(d.actual_context_tokens||d.requested_context_tokens)&&Number.isFinite(d.prefill_tps)&&Number.isFinite(d.decode_tps)); const x=d3.scaleLog().domain(d3.extent(values,d=>d.actual_context_tokens||d.requested_context_tokens)).nice().range([m.l,m.l+w]); const y=d3.scaleLinear().domain([0,d3.max(values,d=>Math.max(d.prefill_tps,d.decode_tps))||1]).nice().range([m.t+h,m.t]); axes(svg,x,y,m,w,h,'контекст, токены','токены/с'); for(const metric of ['prefill_tps','decode_tps']) svg.append('g').selectAll('circle').data(values).join('circle').attr('cx',d=>x(d.actual_context_tokens||d.requested_context_tokens)).attr('cy',d=>y(d[metric])).attr('r',metric==='prefill_tps'?5:3.5).attr('fill',d=>color(d.model||'unknown')).attr('stroke',metric==='prefill_tps'?'#fff':'none').append('title').text(d=>`${{d.model}} · ${{metric}} ${{d[metric].toFixed(2)}} tok/s · ${{d.actual_context_tokens||d.requested_context_tokens}} tokens`); }});
const tbody=document.getElementById('rows'); for(const d of events) {{ const tr=document.createElement('tr'); if(d.status!=='success') tr.className='fail'; const params=JSON.stringify(d.parameters||{{}}); const ref=d.artifact||d.error||''; tr.innerHTML=`<td>${{d.timestamp_utc||''}}</td><td>${{d.model||''}}</td><td>${{d.actual_context_tokens||d.requested_context_tokens||''}}</td><td><code>${{params}}</code></td><td>${{d.prefill_tps??''}}</td><td>${{d.decode_tps??''}}</td><td>${{d.status}}</td><td><code>${{ref}}</code></td>`; tbody.appendChild(tr); }}
</script>
""",
        encoding="utf-8",
    )
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path(__file__).parent / "results" / "benchmark-events.jsonl")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "benchmark-dashboard.html")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(render_dashboard(args.ledger, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
