"""
League dashboard: a small web app showing the live state of the league.

    poetry run python dashboard.py            # then open http://localhost:8765

It only reads the league's files (see leaguefiles.py) and refreshes itself.
"""

import argparse
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import leaguefiles as lf


_reports: dict = {}


def video_reports():
    """The video list with a match report for each (cached per replay)."""
    import showcase_facts as sf

    videos = lf.videos()
    previous = []
    for replay in sorted((lf.LEAGUE_DIR / "replays").glob("round-*.npz")):
        if replay.name not in _reports:
            f = sf.facts(replay)
            _reports[replay.name] = (f, *sf.describe(f, previous or None))
        previous.append(_reports[replay.name][0])
    for v in videos:
        key = f"round-{v['round']:03d}.npz"
        if key in _reports:
            _, v["result"], v["notes"] = _reports[key]
        v["commentary"] = sf.commentary(lf.LEAGUE_DIR, v["round"])
    return videos


def status():
    round_ = lf.latest_round() or {"number": 0, "time": "-", "swiss": [], "h2h": [], "notes": []}
    return {
        "time": time.strftime("%H:%M:%S"),
        "round": round_,
        "champion": lf.champion().stem if lf.champion() else None,
        "crowned": lf.champion_crowned(),
        "events": [{"round": r, "text": t} for r, t in reversed(lf.events(30))],
        "learners": lf.learner_progress(),
        "helpers": lf.helper_progress(),
        "next_round_in": lf.next_round_eta(20),
        "table": lf.football_table(results) if (results := lf.round_results()) else None,
        "table_round": results["round"] if results else None,
        "cpu": lf.compute_usd(),
        "videos": video_reports(),
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Soccer League</title>
<style>
:root { --bg:#0f1512; --card:#17201b; --ink:#e8efe9; --muted:#8fa197; --line:#26332c;
        --blue:#5b7cff; --red:#ff5b5b; --good:#3ecf8e; --bad:#ff7a6b; --accent:#f2c14e; }
@media (prefers-color-scheme: light) {
  :root { --bg:#f4f6f3; --card:#ffffff; --ink:#1c2420; --muted:#5f6f66; --line:#e2e8e4; }
}
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
header { padding:18px 20px 6px; display:flex; flex-wrap:wrap; gap:12px; align-items:baseline; }
h1 { font-size:22px; margin:0; } .sub { color:var(--muted) }
main { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap:14px; padding:14px 20px 30px; }
@media (max-width: 900px) { main { grid-template-columns: minmax(0, 1fr); } }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.card h2 { font-size:15px; margin:0 0 10px; color:var(--muted); font-weight:600; letter-spacing:.02em }
.wide { grid-column: 1 / -1 }
.kpis { display:flex; flex-wrap:wrap; gap:18px }
.kpi b { display:block; font-size:22px } .kpi span { color:var(--muted); font-size:12px }
table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums }
th, td { padding:5px 6px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap }
th:nth-child(2), td:nth-child(2) { text-align:left } th { color:var(--muted); font-weight:500; font-size:12px }
tr.entrant td:nth-child(2) { font-weight:600 } tr.champ td:nth-child(2) { color:var(--accent) }
.bar { position:relative; height:8px; background:var(--line); border-radius:4px; min-width:90px }
.bar i { position:absolute; top:0; bottom:0; border-radius:4px; background:var(--blue) }
.div { position:relative; height:10px; background:var(--line); border-radius:4px; min-width:120px }
.div i { position:absolute; top:0; bottom:0; } .div .mid { left:50%; width:1px; background:var(--muted) }
.pos { color:var(--good) } .neg { color:var(--bad) }
.events li { margin:3px 0; color:var(--ink) } .events { padding-left:18px; margin:0; max-height:360px; overflow:auto }
.tag { display:inline-block; padding:0 6px; border-radius:6px; font-size:11px; background:var(--line); color:var(--muted); margin-right:6px }
.scroll { overflow-x:auto }
video { width:100%; max-width:900px; border-radius:8px; background:#000 }
.vlist { padding-left:18px } .vlist a { color:var(--ink) } .vlist li { margin:0 0 12px } a { color:var(--blue) }
</style></head>
<body>
<header><h1>⚽ AI Soccer League 2: from zero</h1><span class="sub" id="sub">loading…</span></header>
<main>
  <section class="card wide"><div class="kpis" id="kpis"></div></section>
  <section class="card wide"><h2 id="tabletitle">League table</h2>
    <div class="sub" style="margin:-4px 0 8px">Each Swiss pairing is one match: a series of full-length games, scored as the average score per game (e.g. 3.2–2.6). A match is a draw if the teams finish within a quarter of a goal per game. GF/GA add up the match scores.</div>
    <div class="scroll" id="footballwrap"><table id="football"></table></div>
    <h2 style="margin-top:16px">Statistics: every game of the round's Swiss (95% confidence intervals)</h2>
    <div class="scroll" id="swisswrap"><table id="swiss"></table></div></section>
  <section class="card wide"><h2>Showcase videos: the top two of each round, with a match report</h2><div id="videos"></div></section>
  <section class="card"><h2>Head to head with the champion (+ = entrant stronger)</h2><div class="scroll"><table id="h2h"></table></div></section>
  <section class="card"><h2>Style against the champion</h2><div class="scroll"><table id="style"></table></div></section>
  <section class="card"><h2>Learners and helpers</h2><div class="scroll"><table id="learners"></table></div><div id="helpers" class="sub" style="margin-top:8px"></div></section>
  <section class="card"><h2>Events</h2><ul class="events" id="events"></ul></section>
</main>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function fmtEta(s) { if (s == null) return '—'; if (s < 0) return 'playing now'; const m = Math.floor(s/60); return m + ' min ' + Math.floor(s%60) + ' s'; }
function cpuh(d, name) { const h = (d.cpu || {})[name]; return h == null ? '<span class="sub">–</span>' : '$' + h.toFixed(2); }
function render(d) {
  if (d.table) {
    $('tabletitle').textContent = `League table: round ${d.table_round}`;
    $('football').innerHTML = '<tr><th>Pos</th><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th><th>GF</th><th>GA</th><th>GD</th><th>Pts</th><th title="compute used for training so far, at $0.04 per CPU-hour">Compute</th></tr>' +
      d.table.map(t => `<tr class="${t.entrant ? 'entrant' : ''} ${t.name === d.champion ? 'champ' : ''}"><td>${t.pos}</td><td>${t.entrant ? '★ ' : ''}${esc(t.name)}</td>
        <td>${t.p}</td><td>${t.w}</td><td>${t.d}</td><td>${t.l}</td><td>${t.gf.toFixed(1)}</td><td>${t.ga.toFixed(1)}</td>
        <td class="${t.gd >= 0 ? 'pos' : 'neg'}">${t.gd >= 0 ? '+' : ''}${t.gd.toFixed(1)}</td><td><b>${t.pts}</b></td><td>${cpuh(d, t.name)}</td></tr>`).join('');
  } else {
    $('football').innerHTML = '<tr><td class="sub">The first football-style table appears after the next round.</td></tr>';
  }
  const v = d.videos || [];
  if (v.length) {
    const label = x => x.names ? `Round ${x.round}: ${esc(x.names[0])} ${x.score[0]}–${x.score[1]} ${esc(x.names[1])}` : `Round ${x.round}`;
    $('videos').innerHTML = `<ul class="vlist">${v.map(x => `<li><a href="/videos/${x.file}" target="_blank"><b>${label(x)}</b></a>
      ${x.result ? `<div>${esc(x.result)}</div>` : ''}${x.notes ? `<div class="sub">${esc(x.notes)}</div>` : ''}
      ${x.commentary ? `<div><i>${esc(x.commentary)}</i></div>` : ''}</li>`).join('')}</ul>`;
  } else { $('videos').innerHTML = '<span class="sub">No videos yet.</span>'; }
  const r = d.round || {swiss:[], h2h:[], notes:[]};
  $('sub').textContent = `updated ${d.time} · refreshes every 20 s`;
  const top = r.swiss[0];
  $('kpis').innerHTML = [
    ['Round', r.number ?? '—'], ['Played at', r.time ?? '—'], ['Champion', d.champion + (d.crowned ? ` = ${d.crowned[2]} (crowned round ${d.crowned[1]})` : '')],
    ['Top of the Swiss', top ? top.name : '—'], ['Next round', fmtEta(d.next_round_in)],
  ].map(([k, v]) => `<div class="kpi"><span>${k}</span><b>${esc(v)}</b></div>`).join('');
  const maxPts = Math.max(3, ...r.swiss.map(x => x.pts));
  $('swiss').innerHTML = '<tr><th>#</th><th>Brain</th><th>Games</th><th>W-D-L</th><th>GD/game</th><th>Pts/game</th><th title="compute used for training so far, at $0.04 per CPU-hour">Compute</th><th></th></tr>' +
    r.swiss.map(x => `<tr class="${x.entrant ? 'entrant' : ''} ${x.name === d.champion ? 'champ' : ''}">
      <td>${x.rank}</td><td>${x.entrant ? '★ ' : ''}${esc(x.name)}</td><td>${x.games}</td><td>${x.w}-${x.d}-${x.l}</td>
      <td class="${x.gd >= 0 ? 'pos' : 'neg'}">${x.gd >= 0 ? '+' : ''}${x.gd.toFixed(2)} ±${x.gd_ci.toFixed(2)}</td>
      <td>${x.pts.toFixed(2)} ±${x.pts_ci.toFixed(2)}</td><td>${cpuh(d, x.name)}</td>
      <td><div class="bar"><i style="left:0;width:${100 * x.pts / maxPts}%"></i></div></td></tr>`).join('');
  const h = [...r.h2h].sort((a, b) => b.gd - a.gd);
  const noChamp = '<tr><td class="sub">No reigning champion during this round (the first champion is crowned at the end of round 1).</td></tr>';
  $('h2h').innerHTML = '<tr><th></th><th>Entrant</th><th>Games</th><th>W-D-L</th><th>GD/game</th><th>vs champion</th></tr>' +
    h.map(x => { const lo = x.gd - x.gd_ci, hi = x.gd + x.gd_ci, sc = v => 50 + 50 * Math.max(-1, Math.min(1, v / 0.6));
      const sig = lo > 0 ? '▲' : hi < 0 ? '▼' : '';
      return `<tr><td>${sig}</td><td>${esc(x.name)}</td><td>${x.games}</td><td>${x.w}-${x.d}-${x.l}</td>
      <td class="${x.gd >= 0 ? 'pos' : 'neg'}">${x.gd >= 0 ? '+' : ''}${x.gd.toFixed(2)} ±${x.gd_ci.toFixed(2)}</td>
      <td><div class="div"><i class="mid"></i><i style="left:${sc(lo)}%;width:${sc(hi) - sc(lo)}%;background:${x.gd >= 0 ? 'var(--good)' : 'var(--bad)'};opacity:.8"></i></div></td></tr>`; }).join('');
  if (!h.length) { $('h2h').innerHTML = noChamp; $('style').innerHTML = noChamp; }
  else $('style').innerHTML = '<tr><th></th><th>Entrant</th><th>Their half</th><th>Final third</th><th>Shots</th><th>Spread</th><th>Passes</th></tr>' +
    h.map(x => `<tr><td></td><td>${esc(x.name)}</td><td>${(100 * x.style.own_half).toFixed(0)}%</td><td>${(100 * x.style.final_third).toFixed(0)}%</td>
      <td>${x.style.shots.toFixed(1)}</td><td>${x.style.spread.toFixed(0)}</td><td>${x.style.passes.toFixed(1)}</td></tr>`).join('');
  $('learners').innerHTML = '<tr><th></th><th>Learner</th><th>Iterations</th></tr>' +
    Object.entries(d.learners).map(([k, v]) => `<tr><td></td><td>${esc(k)}</td><td>${v}</td></tr>`).join('');
  $('helpers').innerHTML = Object.entries(d.helpers).map(([k, v]) => `<div><b>${esc(k)}</b>: ${esc(v)}</div>`).join('');
  $('events').innerHTML = d.events.map(e => `<li><span class="tag">R${e.round ?? '–'}</span>${esc(e.text)}</li>`).join('');
}
async function tick() { try { render(await (await fetch('/api/status')).json()); } catch (e) { $('sub').textContent = 'cannot reach the dashboard server'; } }
tick(); setInterval(tick, 20000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def send_video(self):
        name = self.path.split("/videos/", 1)[1].split("?")[0]
        path = Path("videos") / lf.LEAGUE_DIR.name / Path(name).name
        if not path.exists() or path.suffix != ".mp4":
            self.send_error(404)
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match:
            start = int(match[1]) if match[1] else 0
            end = int(match[2]) if match[2] else size - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def do_GET(self):
        if self.path.startswith("/videos/"):
            self.send_video()
            return
        if self.path.startswith("/api/status"):
            body, kind = json.dumps(status()).encode(), "application/json"
        else:
            body, kind = PAGE.encode(), "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"League dashboard on http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
