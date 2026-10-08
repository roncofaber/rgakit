"""
review.py
---------
Interactive compound curation review.

Serves a local single-page app for reviewing candidate compounds and
recording keep/remove decisions. Candidates are the union of the given
libraries (deduped by InChIKey), enriched with optional fitted-weight
statistics.

    launch_review("data/libraries/", decisions_path="decisions.json")

Browser keys: y = keep, n = remove (then Enter confirms, Esc cancels),
s/space = skip, arrows = navigate, u = undo last decision. Each decision is
saved immediately to the decisions JSON file. Once the review is done, build
the curated library with ``apply_decisions``.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from collections.abc import Iterable
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .library import SpectraLibrary
from .spectrum import MassSpectrum

DEFAULT_BACKGROUND: tuple[str, ...] = (
    "Hydrogen", "Methane", "Water", "Carbon dioxide",
    "Carbon monoxide", "Ammonia", "Oxygen", "Nitrogen",
)

WEIGHT_EPS = 1e-6

_lock = threading.Lock()


#%% Inputs

def _iter_spectra(libraries) -> Iterable[MassSpectrum]:
    if isinstance(libraries, SpectraLibrary):
        yield from libraries
    elif isinstance(libraries, MassSpectrum):
        yield libraries
    elif isinstance(libraries, (str, Path)):
        path = Path(libraries)
        if path.is_dir():
            for p in sorted(path.glob("*.pkl")):
                yield from SpectraLibrary.load(p)
        elif path.exists():
            yield from SpectraLibrary.load(path)
        else:
            raise FileNotFoundError(path)
    elif isinstance(libraries, Iterable):
        for item in libraries:
            yield from _iter_spectra(item)
    else:
        raise TypeError(f"Cannot build candidates from {type(libraries).__name__}")


def load_stats_csv(path, name_col: str = "name", n_col: str = "n_samples",
                   mean_col: str = "mean_w", max_col: str = "max_w") -> dict[str, dict]:
    """
    Load pre-computed occurrence statistics keyed by compound name.

    Expected CSV columns: name, n_samples, mean_w, max_w.
    """
    import csv

    stats = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            stats[row[name_col]] = {
                "n_samples": int(row[n_col]),
                "mean_w": float(row[mean_col]),
                "max_w": float(row[max_col]),
            }
    return stats


#%% Candidates

def build_candidates(libraries, *, weights: dict[str, list[float]] | None = None,
                     stats: dict[str, dict] | None = None,
                     background_names: Iterable[str] = DEFAULT_BACKGROUND) -> list[dict]:
    """
    Build the review candidate list from libraries.

    Parameters
    ----------
    libraries : SpectraLibrary | MassSpectrum | path | iterable thereof
        Sources for the candidate union. A directory is scanned for .pkl
        library files.
    weights : dict {compound name: list of fitted weights}, optional
        Per-sample fitted weights; used to compute occurrence statistics.
    stats : dict {compound name: {n_samples, mean_w, max_w}}, optional
        Pre-computed statistics (takes precedence over *weights*).
    background_names : iterable of str
        Compound names flagged as background tier.

    Returns
    -------
    list of dicts, sorted with background compounds first, then by descending
    occurrence, then never-fitted ones alphabetically.
    """
    union = {}
    for spec in _iter_spectra(libraries):
        ik = spec.metadata.get("inchikey")
        if ik:
            union[ik] = spec

    bg = set(background_names)
    compounds = []
    for ik, spec in union.items():
        meta = spec.metadata
        s = (stats or {}).get(spec.name)
        if s is None and weights:
            nz = [v for v in weights.get(spec.name, []) if v > WEIGHT_EPS]
            s = {
                "n_samples": len(nz),
                "mean_w": sum(nz) / len(nz) if nz else 0.0,
                "max_w": max(nz) if nz else 0.0,
            }
        if s is None:
            s = {"n_samples": 0, "mean_w": 0.0, "max_w": 0.0}

        compounds.append({
            "name": spec.name,
            "inchikey": ik,
            "formula": meta.get("formula", ""),
            "cas": meta.get("cas", ""),
            "mw": meta.get("mw"),
            "smiles": meta.get("smiles", ""),
            "n_samples": s["n_samples"],
            "mean_w": s["mean_w"],
            "max_w": s["max_w"],
        })

    def sort_key(c):
        tier = 0 if c["name"] in bg else (1 if c["n_samples"] else 2)
        return (tier, -c["n_samples"], c["name"])

    compounds.sort(key=sort_key)
    return compounds


#%% Compound details

def _mol_svg(smiles: str) -> str | None:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    d = rdMolDraw2D.MolDraw2DSVG(340, 340)
    o = d.drawOptions()
    o.backgroundColour = (0.09, 0.10, 0.13, 1.0)
    o.bondLineWidth = 2
    o.padding = 0.12
    o.updateAtomPalette({6: (0.85, 0.87, 0.92), 1: (0.60, 0.62, 0.66)})
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return d.GetDrawingText()


def _spectrum_json(spec: MassSpectrum) -> dict:
    import numpy as np

    mz, inten = spec.mz, spec.intensity
    if len(mz) == 0:
        return {"mz": [], "rel": []}
    top = np.argsort(-inten)[:50]
    sel = top[np.argsort(mz[top])]
    mx = float(inten.max())
    return {
        "mz": [int(m) for m in mz[sel]],
        "rel": [round(float(i) / mx, 4) for i in inten[sel]],
    }


#%% Decisions

def load_decisions(path) -> dict:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_decisions(path, dec: dict) -> None:
    Path(path).write_text(json.dumps(dec, indent=1) + "\n")


def apply_decisions(libraries, decisions_path, *, verdict: str = "remove",
                    background_names: Iterable[str] = DEFAULT_BACKGROUND) -> SpectraLibrary:
    """
    Build a curated SpectraLibrary from libraries and a decisions JSON file.

    Compounds whose recorded verdict matches *verdict* are dropped.
    """
    dec = load_decisions(decisions_path)
    spectra = [s for s in _iter_spectra(libraries)
               if dec.get(s.metadata.get("inchikey"), {}).get("verdict") != verdict]
    return SpectraLibrary(spectra)


#%% HTTP server

def _make_handler(union: dict, candidates: list[dict], decisions_path, bg_names: list[str]):

    class Handler(BaseHTTPRequestHandler):

        def log_message(self, fmt, *args):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, text):
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._html(PAGE)
            elif self.path == "/api/list":
                self._json(candidates)
            elif self.path == "/api/meta":
                self._json({"background": bg_names})
            elif self.path == "/api/decisions":
                with _lock:
                    self._json(load_decisions(decisions_path))
            elif self.path.startswith("/api/compound/"):
                ik = self.path.rsplit("/", 1)[1]
                spec = union.get(ik)
                if spec is None:
                    self._json({"error": "unknown inchikey"}, 404)
                    return
                self._json({
                    "svg": _mol_svg(spec.metadata.get("smiles", "")),
                    "spectrum": _spectrum_json(spec),
                })
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return

            if self.path == "/api/decide":
                ik = payload.get("inchikey")
                v = payload.get("verdict")
                if ik not in union or v not in ("keep", "remove"):
                    self._json({"error": "bad request"}, 400)
                    return
                with _lock:
                    dec = load_decisions(decisions_path)
                    dec[ik] = {
                        "name": union[ik].name,
                        "verdict": v,
                        "reason": payload.get("reason", ""),
                        "decided_at": date.today().isoformat(),
                    }
                    save_decisions(decisions_path, dec)
                self._json({"ok": True})

            elif self.path == "/api/undo":
                with _lock:
                    dec = load_decisions(decisions_path)
                    if not dec:
                        self._json({"ok": False})
                        return
                    ik = next(reversed(dec))
                    entry = dec.pop(ik)
                    save_decisions(decisions_path, dec)
                self._json({"ok": True, "inchikey": ik, "entry": entry})

            else:
                self.send_error(404)

    return Handler


#%% Front end

PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Compound curation review</title>
<style>
  :root { --bg:#16181d; --card:#1f232b; --panel:#171a20; --fg:#d7dae0;
          --dim:#8a919c; --keep:#4caf7d; --rem:#e06c6c; --acc:#61afef;
          --peak:#53d6e8; }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--fg); margin: 0;
         font: 15px/1.45 -apple-system, "Segoe UI", sans-serif; }
  header { display: flex; align-items: center; gap: 16px; padding: 10px 18px 8px;
           background: var(--card); border-bottom: 1px solid #2c313a;
           position: sticky; top: 0; z-index: 5; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  #progress { color: var(--dim); font-variant-numeric: tabular-nums; }
  .spacer { flex: 1; }
  label.tog { color: var(--dim); cursor: pointer; user-select: none; }
  #pbar { width: 100%; height: 3px; background: #262b34; border-radius: 2px;
          margin-top: 2px; }
  #pfill { height: 100%; width: 0%; background: var(--acc); border-radius: 2px;
           transition: width .3s; }
  main { max-width: 1100px; margin: 0 auto; padding: 20px 22px; }
  .row1 { display: flex; gap: 22px; }
  .struct { flex: 0 0 340px; background: var(--panel); border: 1px solid #2c313a;
            border-radius: 10px; min-height: 340px; display: flex;
            align-items: center; justify-content: center; overflow: hidden; }
  .struct svg { max-width: 100%; }
  .struct .none { color: var(--dim); padding: 20px; text-align: center; }
  .info { flex: 1; min-width: 0; }
  .info h2 { margin: 0 0 4px; font-size: 22px; }
  .badges { margin: 6px 0 14px; }
  .badge { display: inline-block; padding: 2px 9px; border-radius: 20px;
           font-size: 12px; margin-right: 6px; background: #2c313a; color: var(--dim); }
  .badge.bg   { background: #3a3325; color: #d9b96c; }
  .badge.prod { background: #243540; color: var(--acc); }
  .badge.seen { background: #2c3a2c; color: var(--keep); }
  .badge.kept { background: #1e4030; color: var(--keep); }
  .badge.rem  { background: #40262a; color: var(--rem); }
  .badge.pending { background: #40262a; color: var(--rem); animation: pulse 1s infinite; }
  @keyframes pulse { 50% { opacity: .55; } }
  dl { display: grid; grid-template-columns: 110px minmax(0, 1fr); gap: 4px 14px;
       margin: 0 0 16px; align-items: baseline; }
  dt { color: var(--dim); text-align: right; }
  dd { margin: 0; min-width: 0; }
  dd code { font: 13px/1.5 ui-monospace, Menlo, monospace; color: #b8c4d4;
            background: #262b34; padding: 5px 9px; border-radius: 6px;
            display: inline-block; word-break: break-all; max-width: 100%; }
  .stats { display: flex; gap: 26px; margin: 0 0 16px; padding: 10px 14px;
           background: var(--panel); border: 1px solid #2c313a; border-radius: 8px; }
  .stats .st { display: flex; flex-direction: column; }
  .stats .st b { font-size: 17px; font-variant-numeric: tabular-nums; }
  .stats .st span { color: var(--dim); font-size: 12px; }
  .stats .st.warn b { color: var(--dim); }
  #reason { width: 100%; background: #2a303b; color: var(--fg); border: 1px solid #3d4655;
            border-radius: 6px; padding: 7px 10px; margin: 0 0 12px; }
  #reason:focus { outline: none; border-color: var(--rem); }
  .btns { display: flex; gap: 12px; }
  button { font-size: 15px; padding: 10px 20px; border-radius: 8px; cursor: pointer;
           border: 1px solid #3a4150; background: #262b34; color: var(--fg); }
  button:hover { filter: brightness(1.2); }
  button.keep { background: #1e4030; border-color: var(--keep); }
  button.rem  { background: #40262a; border-color: var(--rem); }
  #spectrumbox { margin-top: 18px; background: var(--panel); border: 1px solid #2c313a;
                  border-radius: 10px; padding: 10px 12px 6px; }
  #spectrum { width: 100%; height: 260px; display: block; }
  #status { position: fixed; bottom: 14px; right: 18px; color: var(--dim);
            font-size: 13px; }
  .done { margin: 80px auto; text-align: center; }
  .done h2 { font-size: 26px; }
  @media (max-width: 900px) {
    .row1 { flex-direction: column; }
    .struct { flex: none; }
  }
</style>
</head>
<body>
<header>
  <h1>Compound curation</h1>
  <span id="progress"></span>
  <div class="spacer"></div>
  <label class="tog"><input type="checkbox" id="tog" checked> only undecided</label>
  <button id="undo" title="u">Undo (u)</button>
  <div id="pbar"><div id="pfill"></div></div>
</header>
<main id="main"></main>
<div id="status"></div>
<script>
let comps = [], dec = {}, idx = 0, onlyUndecided = true;
let bgNames = [], pending = null;
const detailCache = {};
let renderToken = 0;

const filtered = () =>
  comps.map(c => c.inchikey).filter(ik => !onlyUndecided || !dec[ik]);

async function init() {
  comps = await (await fetch('/api/list')).json();
  dec   = await (await fetch('/api/decisions')).json();
  bgNames = (await (await fetch('/api/meta')).json()).background;
  document.getElementById('tog').onchange = e => { onlyUndecided = e.target.checked; idx = 0; render(); };
  document.getElementById('undo').onclick = undo;
  document.addEventListener('keydown', onKey);
  render();
}

function onKey(e) {
  const typing = e.target.tagName === 'INPUT' && e.target.type === 'text';
  if (typing) {
    if (e.key === 'Enter' && pending) submitPending();
    else if (e.key === 'Escape' && pending) { pending = null; render(); }
    return;
  }
  if (e.key === 'y' || e.key === 'Y') decide('keep');
  else if (e.key === 'n' || e.key === 'N') startRemove();
  else if (e.key === 's' || e.key === ' ' || e.key === 'ArrowRight') next(1);
  else if (e.key === 'ArrowLeft') next(-1);
  else if (e.key === 'u' || e.key === 'U') undo();
}

function next(step) {
  const f = filtered();
  if (!f.length) return render();
  idx = Math.min(Math.max(idx + step, 0), f.length - 1);
  render();
}

async function decide(verdict) {
  const f = filtered();
  if (!f.length) return render();
  const ik = f[idx];
  const reason = document.getElementById('reason')?.value || "";
  await fetch('/api/decide', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({inchikey: ik, verdict, reason})
  });
  dec[ik] = {verdict};
  pending = null;
  render();
}

function startRemove() {
  pending = 'remove';
  render();
  document.getElementById('reason')?.focus();
}

async function submitPending() {
  if (!pending) return;
  await decide(pending);
}

async function undo() {
  const r = await (await fetch('/api/undo', {method: 'POST'})).json();
  if (!r.ok) return;
  delete dec[r.inchikey];
  const f = filtered();
  const pos = f.indexOf(r.inchikey);
  if (pos >= 0) idx = pos;
  render();
}

async function render() {
  const token = ++renderToken;
  const f = filtered();
  const done = comps.filter(c => dec[c.inchikey]).length;
  document.getElementById('progress').textContent =
    `${done} decided / ${comps.length}`;
  document.getElementById('pfill').style.width =
    `${100 * done / Math.max(comps.length, 1)}%`;

  const main = document.getElementById('main');
  if (!f.length) {
    const kept = comps.filter(c => dec[c.inchikey]?.verdict === 'keep').length;
    const rem  = comps.filter(c => dec[c.inchikey]?.verdict === 'remove').length;
    main.innerHTML = `<div class="done"><h2>Review complete</h2>
      <p>${done} decisions: ${kept} kept, ${rem} marked for removal.</p>
      <p>Uncheck "only undecided" to revisit.</p></div>`;
    return;
  }

  idx = Math.min(idx, f.length - 1);
  const c = comps.find(x => x.inchikey === f[idx]);
  const d = dec[c.inchikey];
  const badges = [
    isBackground(c) ? '<span class="badge bg">background</span>'
                    : '<span class="badge prod">product</span>',
    c.n_samples ? '<span class="badge seen">seen in fits</span>'
                : '<span class="badge">never fitted</span>',
  ];
  if (pending === 'remove') badges.push('<span class="badge pending">REMOVE - Enter to confirm, Esc to cancel</span>');
  else if (d?.verdict === 'keep') badges.push('<span class="badge kept">marked: KEEP</span>');
  else if (d?.verdict === 'remove') badges.push('<span class="badge rem">marked: REMOVE</span>');

  const fmt = v => v ? v.toExponential(2) : '—';
  main.innerHTML = `
    <div class="row1">
      <div class="struct" id="struct"><span class="none">loading…</span></div>
      <div class="info">
        <h2>${esc(c.name)}</h2>
        <div class="badges">${badges.join('')}</div>
        <dl>
          <dt>Formula</dt><dd>${esc(c.formula)}</dd>
          <dt>CAS</dt><dd>${esc(c.cas || '—')}</dd>
          <dt>MW</dt><dd>${c.mw ?? '—'}</dd>
          <dt>SMILES</dt><dd><code>${esc(c.smiles)}</code></dd>
          <dt>InChIKey</dt><dd>${esc(c.inchikey)}</dd>
        </dl>
        <div class="stats">
          <div class="st ${c.n_samples ? '' : 'warn'}"><b>${c.n_samples}</b><span>samples fitted</span></div>
          <div class="st ${c.n_samples ? '' : 'warn'}"><b>${c.n_samples ? fmt(c.mean_w) : '—'}</b><span>mean weight</span></div>
          <div class="st ${c.n_samples ? '' : 'warn'}"><b>${c.n_samples ? fmt(c.max_w) : '—'}</b><span>max weight</span></div>
        </div>
        <input id="reason" type="text" placeholder="reason (optional) - Enter confirms a removal">
        <div class="btns">
          <button class="keep" onclick="decide('keep')">Keep (y)</button>
          <button class="rem"  onclick="startRemove()">Remove (n)</button>
          <button onclick="next(1)">Skip (s)</button>
          <button onclick="next(-1)">Prev (←)</button>
        </div>
      </div>
    </div>
    <div id="spectrumbox"><canvas id="spectrum" width="1040" height="260"></canvas></div>`;
  document.getElementById('status').textContent = `${idx + 1} / ${f.length}`;

  if (!detailCache[c.inchikey]) {
    detailCache[c.inchikey] = await (await fetch('/api/compound/' + c.inchikey)).json();
  }
  if (token !== renderToken) return;
  const det = detailCache[c.inchikey];
  const box = document.getElementById('struct');
  if (det.svg) { box.innerHTML = det.svg; }
  else { box.innerHTML = `<span class="none">no structure<br>${esc(c.smiles)}</span>`; }
  drawSpectrum(det.spectrum);
}

function isBackground(c) {
  return bgNames.includes(c.name);
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"]/g, m =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m]));
}

function drawSpectrum(sp) {
  const cv = document.getElementById('spectrum');
  if (!cv || !sp.mz.length) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, padL = 44, padR = 10, padB = 34, padT = 10;
  ctx.fillStyle = '#14161b'; ctx.fillRect(0, 0, W, H);
  const mzMax = Math.max(20, Math.ceil(Math.max(...sp.mz) / 10) * 10);
  const X = m => padL + (m / mzMax) * (W - padL - padR);
  const Y = r => H - padB - r * (H - padB - padT);
  ctx.strokeStyle = '#2c313a'; ctx.fillStyle = '#a8b0bb';
  ctx.font = '12px sans-serif'; ctx.lineWidth = 1;
  for (let t = 0; t <= mzMax; t += mzMax > 150 ? 50 : 20) {
    ctx.beginPath(); ctx.moveTo(X(t), padT); ctx.lineTo(X(t), H - padB); ctx.stroke();
    ctx.fillText(t, X(t) - 8, H - padB + 16);
  }
  ctx.fillText('1.0', 12, padT + 4);
  ctx.fillText('0.5', 12, (H - padB + padT) / 2 + 4);
  ctx.beginPath(); ctx.moveTo(padL, Y(0.5)); ctx.lineTo(W - padR, Y(0.5));
  ctx.setLineDash([3, 4]); ctx.stroke(); ctx.setLineDash([]);
  ctx.fillText('m/z', (padL + W) / 2 - 12, H - 4);
  ctx.save();
  ctx.translate(12, (H - padB + padT) / 2 - 10);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('rel. intensity', 0, 0);
  ctx.restore();
  ctx.strokeStyle = '#53d6e8'; ctx.lineWidth = 2.2;
  ctx.beginPath();
  for (let i = 0; i < sp.mz.length; i++) {
    ctx.moveTo(X(sp.mz[i]), H - padB);
    ctx.lineTo(X(sp.mz[i]), Y(sp.rel[i]));
  }
  ctx.stroke();
}

init();
</script>
</body>
</html>
"""


#%% Public API

def launch_review(libraries, *, port: int = 8765,
                  decisions_path = "curated_decisions.json",
                  weights: dict[str, list[float]] | None = None,
                  stats: dict[str, dict] | None = None,
                  background_names: Iterable[str] = DEFAULT_BACKGROUND,
                  candidates_path = None,
                  rebuild: bool = False,
                  open_browser: bool = True) -> None:
    """
    Launch the interactive compound curation review app (blocking).

    Parameters
    ----------
    libraries : SpectraLibrary | MassSpectrum | path | iterable thereof
        Sources for the candidate union. A directory is scanned for .pkl
        library files.
    port : TCP port for the local server.
    decisions_path : JSON file where keep/remove verdicts are stored.
    weights : dict {compound name: list of fitted weights}, optional
        Per-sample fitted weights used for occurrence statistics.
    stats : dict {compound name: {n_samples, mean_w, max_w}}, optional
        Pre-computed statistics (takes precedence over *weights*).
    background_names : compound names flagged as background tier.
    candidates_path : optional cache file for the candidate list.
    rebuild : force recomputation of the candidate list.
    open_browser : open the app in the default browser.
    """
    candidates = None
    if candidates_path is not None and Path(candidates_path).exists() and not rebuild:
        candidates = json.loads(Path(candidates_path).read_text())
    if candidates is None:
        candidates = build_candidates(libraries, weights=weights, stats=stats,
                                      background_names=background_names)
        if candidates_path is not None:
            Path(candidates_path).write_text(json.dumps(candidates, indent=1))

    union = {}
    for spec in _iter_spectra(libraries):
        ik = spec.metadata.get("inchikey")
        if ik:
            union[ik] = spec

    handler = _make_handler(union, candidates, Path(decisions_path),
                            list(background_names))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}"
    print(f"{len(candidates)} candidate compounds")
    print(f"Review app: {url}   (Ctrl-C to stop)")
    print(f"Decisions:  {Path(decisions_path).resolve()}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


#%% CLI

def cli_main() -> None:
    ap = argparse.ArgumentParser(
        prog="rgakit-review",
        description="Interactive compound curation review for spectral libraries.")
    ap.add_argument("libraries", nargs="+",
                    help="Library file(s), or directory of .pkl library files")
    ap.add_argument("-p", "--port", type=int, default=8765)
    ap.add_argument("-d", "--decisions", default="curated_decisions.json",
                    help="Decisions JSON file (default: ./curated_decisions.json)")
    ap.add_argument("--stats", help="CSV with columns: name, n_samples, mean_w, max_w")
    ap.add_argument("--background", nargs="*", default=list(DEFAULT_BACKGROUND),
                    help="Compound names to flag as background tier")
    ap.add_argument("--candidates", help="Cache file for the candidate list")
    ap.add_argument("--rebuild", action="store_true",
                    help="Rebuild the cached candidate list")
    ap.add_argument("--no-open", action="store_true", help="Do not open the browser")
    args = ap.parse_args()

    stats = load_stats_csv(args.stats) if args.stats else None
    launch_review(args.libraries, port=args.port, decisions_path=args.decisions,
                  stats=stats, background_names=args.background,
                  candidates_path=args.candidates, rebuild=args.rebuild,
                  open_browser=not args.no_open)


if __name__ == "__main__":
    cli_main()
