"""MBAX 6418 - Assignment 1, Step 3: generate a self-contained HTML results
dashboard from Step 2's saved CSV (step2_results.csv).

No LLM calls are made: the CSV is the single source of truth. The generator
computes every figure from the CSV and embeds both the stats and the review
rows as JSON inside the HTML, which renders charts and the review table with
vanilla JavaScript (inline CSS, hand-rolled SVG, zero external requests --
fully offline and self-contained).

Usage:
    python step3_dashboard.py          # writes step3_dashboard.html

Output:
    step3_dashboard.html  -- open in any browser, or preview here.
"""

import csv
import datetime as _dt
import json
import os
import sys

from review_prompt import load_env_file

CSV_PATH = "step2_results.csv"
OUT_PATH = "step3_dashboard.html"

# Provenance metadata (non-numeric): model/endpoint as used for the Step 2
# run, read from .env so the dashboard always reflects the real config.
load_env_file()
MODEL_NAME = os.environ.get("MODEL", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
ENDPOINT = os.environ.get("OPENAI_BASE_URL", "").rstrip("/") or "http://dobolyi.com:9001/v1"


def pct(part: int, whole: int) -> str:
    """Percentage as a display string, e.g. 98.0%."""
    return f"{round(100 * part / whole, 1):.1f}%" if whole else "n/a"


def pct_num(part: int, whole: int) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


def read_results(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        raw = list(csv.DictReader(fh))
    rows = []
    for r in raw:
        rows.append({
            "index": int(r["index"]),
            "asin": r["asin"],
            "title": r["title"],
            "text": r["text"],
            "rating": float(r["rating"]),
            "correct_label": r["correct_label"],
            "predicted_label": r["predicted_label"],
            "correct": r["correct"] == "True",
            "raw_output": r["raw_output"],
        })
    return rows


def compute_stats(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(r["correct"] for r in rows)
    incorrect = total - correct
    stats = {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "acc": pct(correct, total),
        "acc_num": pct_num(correct, total),
        "inc": pct(incorrect, total),
        "inc_num": pct_num(incorrect, total),
        "classes": {},
        "ratings": [],
        "mismatch_indices": [r["index"] for r in rows if not r["correct"]],
    }
    for label in ("POSITIVE", "NEGATIVE"):
        sub = [r for r in rows if r["correct_label"] == label]
        n, ok = len(sub), sum(r["correct"] for r in sub)
        stats["classes"][label] = {
            "count": n, "correct": ok, "incorrect": n - ok,
            "acc": pct(ok, n), "err": pct(n - ok, n),
            "acc_num": pct_num(ok, n), "err_num": pct_num(n - ok, n),
        }
    for rv in range(1, 6):
        stats["ratings"].append({
            "rating": rv,
            "count": sum(1 for r in rows if int(r["rating"]) == rv),
        })
    return stats


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MBAX 6418 · Assignment 1 · Review Classification Dashboard</title>
<style>
  :root{
    --paper:#F5F2EB; --card:#FFFFFF; --line:#E4DFD3; --ink:#1B1915; --mut:#6F6A5E;
    --gold:#C9A227; --gold-soft:#F3EAC8;
    --good:#1F6F52; --good-bg:#E5F1EB;
    --bad:#B4472C;  --bad-bg:#F9E9E3;
    --pos:#2D7D5F;  --pos-bg:#E4F1EB;
    --neg:#B5542F;  --neg-bg:#F7E8E1;
    --serif:"Iowan Old Style","Palatino Linotype","Book Antiqua",Palatino,Georgia,serif;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --shadow:0 1px 2px rgba(27,25,21,.05),0 10px 28px -14px rgba(27,25,21,.16);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink);
       font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
  ::selection{background:var(--gold-soft)}

  /* Masthead */
  .masthead{background:var(--ink);color:#fff;border-bottom:4px solid var(--gold)}
  .mh-inner{max-width:1180px;margin:0 auto;padding:34px 28px 26px;
            display:flex;justify-content:space-between;align-items:flex-end;gap:20px;flex-wrap:wrap}
  .kicker{font:600 11px/1.2 var(--sans);letter-spacing:.22em;text-transform:uppercase;
          color:var(--gold);margin:0 0 10px}
  .masthead h1{margin:0;font:600 clamp(26px,4vw,36px)/1.12 var(--serif);letter-spacing:.01em}
  .sub{margin:8px 0 0;color:#B9B2A4;font-size:13.5px}
  .mh-meta{display:flex;gap:8px;flex-wrap:wrap}
  .chip{display:inline-block;padding:3px 9px;border-radius:999px;
        font:600 11px/1.4 var(--sans);letter-spacing:.02em;white-space:nowrap}
  .chip.src{color:var(--ink);background:var(--gold-soft);border:1px solid var(--gold)}
  .chip.model{color:#CFC8B7;border:1px solid #3D3930;background:#26231D}
  .chip.pos,.chip.good{color:var(--pos);background:var(--pos-bg)}
  .chip.neg{color:var(--neg);background:var(--neg-bg)}
  .chip.bad{color:var(--bad);background:var(--bad-bg)}
  .chip.warn{outline:1px dashed var(--bad);outline-offset:-1px}

  main{max-width:1180px;margin:0 auto;padding:26px 28px 8px}

  /* KPI band */
  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:16px;margin-bottom:20px}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);
       padding:18px 20px 16px;position:relative;overflow:hidden}
  .kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--gold)}
  .kpi-label{font:600 10.5px/1.2 var(--sans);letter-spacing:.16em;text-transform:uppercase;
             color:var(--mut);margin:0 0 8px}
  .kpi-value{font:600 34px/1.05 var(--serif);margin:0;letter-spacing:-.01em}
  .kpi-sub{margin:8px 0 0;font-size:12px;color:var(--mut)}

  /* Cards */
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
        box-shadow:var(--shadow);padding:20px 22px}
  .card.full,.mismatches,.reviews{grid-column:1/-1}
  h2{font:600 11px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--mut);
     margin:0 0 16px;display:flex;align-items:center;gap:9px}
  h2::before{content:"";width:14px;height:2px;background:var(--gold)}
  .count-badge{margin-left:auto;background:var(--ink);color:#fff;border-radius:999px;
               padding:3px 10px;font:600 11px/1.3 var(--sans)}

  /* Horizontal bars */
  .legend{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:10px}
  .lg{font:600 12.5px/1.3 var(--sans)}
  .lg::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}
  .lg.good::before{background:var(--good)} .lg.bad::before{background:var(--bad)}
  .lg.pos::before{background:var(--pos)} .lg.neg::before{background:var(--neg)}
  .hbar{display:flex;height:34px;border-radius:9px;overflow:hidden;background:var(--line);gap:2px}
  .hbar .seg{display:block;height:100%}
  .hbar .seg.good{background:var(--good)} .hbar .seg.bad{background:var(--bad)}
  .hbar .seg.gold{background:var(--gold)} .hbar .seg.pos{background:var(--pos)} .hbar .seg.neg{background:var(--neg)}
  .hbar.small{height:12px;border-radius:6px;flex:1}

  .cls{margin-bottom:14px}
  .cls:last-child{margin-bottom:0}
  .cls-head{display:flex;justify-content:space-between;gap:10px;margin-bottom:7px;flex-wrap:wrap}
  .cls-name{font:700 13px/1.2 var(--sans);letter-spacing:.04em}
  .cls-nums{font:600 12px/1.3 var(--sans);color:var(--mut)}
  .note{font-size:12.5px;color:var(--mut);margin:12px 0 0}

  /* Donut + mix legend */
  #mix{display:flex;align-items:center;gap:24px;flex-wrap:wrap}
  .donut-num{font:600 26px/1 var(--serif);fill:var(--ink)}
  .donut-cap{font:600 10px/1 var(--sans);letter-spacing:.14em;fill:var(--mut);text-transform:uppercase}
  .mix-legend{display:flex;flex-direction:column;gap:9px}

  /* Rating rows */
  .rrow{display:flex;align-items:center;gap:12px;margin-bottom:9px}
  .rlab{width:104px;color:#8A7F45;font-size:14px;letter-spacing:2px;white-space:nowrap}
  .rlab .off{color:var(--line)}
  .rcount{width:34px;text-align:right;font:600 13px/1 var(--sans);font-variant-numeric:tabular-nums}

  /* Mismatch cards */
  .mm{border-left:3px solid var(--bad);background:#FDF8F5;border-radius:10px;
      padding:14px 18px;margin-bottom:12px}
  .mm:last-child{margin-bottom:0}
  .mm-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}
  .mm-idx{font:600 11px/1.2 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--mut)}
  .mm-rating{color:#8A7F45;font-size:13px;letter-spacing:2px}
  .mm-title{font:600 17px/1.35 var(--serif);margin:0 0 6px}
  .mm-text{margin:0;font-size:14px;color:#4A463D}
  .mm-raw{margin:8px 0 0;font:11.5px/1.5 ui-monospace,Consolas,monospace;color:var(--mut);
          background:var(--card);border:1px solid var(--line);border-radius:6px;padding:7px 10px}

  /* Table */
  .table-tools{display:flex;justify-content:space-between;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
  .fchip{border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:999px;
         padding:5px 12px;font:600 12px/1.2 var(--sans);cursor:pointer;transition:all .12s}
  .fchip b{font-variant-numeric:tabular-nums}
  .fchip:hover{border-color:var(--gold)}
  .fchip.on{background:var(--ink);color:#fff;border-color:var(--ink)}
  #search{border:1px solid var(--line);background:var(--paper);border-radius:999px;
          padding:7px 14px;font:13px var(--sans);min-width:230px;color:var(--ink)}
  #search:focus{outline:none;border-color:var(--gold);background:#fff}
  .table-wrap{max-height:65vh;overflow:auto;border:1px solid var(--line);border-radius:10px}
  table{border-collapse:collapse;width:100%;font-size:13px}
  thead th{position:sticky;top:0;background:#FBF9F4;z-index:2;text-align:left;cursor:pointer;
           font:600 10.5px/1.2 var(--sans);letter-spacing:.12em;text-transform:uppercase;color:var(--mut);
           padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap;user-select:none}
  thead th[data-key]::after{content:"↕";margin-left:5px;color:#C7C1B2}
  thead th[data-dir="asc"]::after{content:"↑";color:var(--gold)}
  thead th[data-dir="desc"]::after{content:"↓";color:var(--gold)}
  tbody td{padding:9px 12px;border-bottom:1px solid #EFEBE1;vertical-align:top}
  tbody tr:last-child td{border-bottom:none}
  tbody tr:hover{background:#FBF8F1}
  tbody tr.mm-row{background:#FDF6F2}
  tbody tr.mm-row td:first-child{box-shadow:inset 3px 0 0 var(--bad)}
  tbody tr.open{background:#FFFDF6}
  td.num{font-variant-numeric:tabular-nums;white-space:nowrap;color:var(--mut)}
  .t-title{font-weight:600;min-width:210px;max-width:260px}
  .t-text{min-width:260px;max-width:420px;color:#4A463D;cursor:pointer}
  .t-text .t-full{color:var(--mut)}
  tr:not(.open) .t-text .t-full{display:none;}

  .foot{max-width:1180px;margin:26px auto 40px;padding:0 28px;color:var(--mut);
        font-size:12px;border-top:1px solid var(--line);padding-top:16px}
  .foot b{color:var(--ink)}

  @media (max-width:820px){ .grid{grid-template-columns:1fr} }
</style>
</head>
<body>

<header class="masthead">
  <div class="mh-inner">
    <div>
      <p class="kicker">MBAX 6418 · Assignment 1 · Step 3</p>
      <h1>Review Classification Dashboard</h1>
      <p class="sub">Amazon 2023 Gift Cards · 100-review sample ·
         LLM sentiment labels vs. rating-derived labels · ratings never shown to the model</p>
    </div>
    <div class="mh-meta">
      <span class="chip src">source: step2_results.csv</span>
      <span class="chip model" id="chip-model">model</span>
    </div>
  </div>
</header>

<main>
  <section class="kpis" id="kpis" aria-label="Key figures"></section>

  <section class="grid">
    <article class="card full">
      <h2>Correct vs. incorrect</h2>
      <div id="outcome"></div>
    </article>

    <article class="card">
      <h2>Outcome by class</h2>
      <div id="byclass"></div>
      <p class="note" id="byclass-note"></p>
    </article>

    <article class="card">
      <h2>Class mix of the sample</h2>
      <div id="mix"></div>
      <p class="note">Labels are derived from the rating (rating ≥ 4 → POSITIVE), so
         the 93:7 split follows the rating distribution — the sample is heavily
         imbalanced toward positive reviews.</p>
    </article>

    <article class="card full">
      <h2>Rating distribution</h2>
      <div id="ratings" style="max-width:560px"></div>
    </article>
  </section>

  <section class="card mismatches" id="mismatches"></section>

  <section class="card reviews">
    <h2>Review-level results <span class="count-badge" id="table-count"></span></h2>
    <div class="table-tools">
      <div class="chips" id="filters"></div>
      <input id="search" type="search" placeholder="Search title, text, or asin …"
             aria-label="Search reviews">
    </div>
    <div class="table-wrap">
      <table id="tbl">
        <thead><tr>
          <th data-key="index" data-dir="asc">index</th>
          <th data-key="rating">rating</th>
          <th data-key="correct_label">expected</th>
          <th data-key="predicted_label">predicted</th>
          <th data-key="correct">status</th>
          <th>title</th>
          <th>text <span style="font-weight:400;text-transform:none;letter-spacing:0">(click a row to expand)</span></th>
        </tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
  </section>
</main>

<footer class="foot" id="foot"></footer>

<script id="dash-data" type="application/json">__DATA__</script>
<script>
(() => {
  const DATA = JSON.parse(document.getElementById('dash-data').textContent);
  const { meta, stats, rows } = DATA;
  const $ = s => document.querySelector(s);
  const esc = s => String(s ?? '').replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const stars = r => { const n = Math.max(1, Math.min(5, Math.round(r)));
    return '★'.repeat(n) + '<span class="off">' + '☆'.repeat(5 - n) + '</span>'; };
  const seg = (w, c) => `<span class="seg ${c}" style="width:${w}%"></span>`;
  const shorten = (s, n) => s && s.length > n ? s.slice(0, n - 1) + '…' : s;

  /* ---- KPI cards ---- */
  const kpis = [
    { l: 'Overall accuracy', v: stats.acc, s: `${stats.correct} of ${stats.total} reviews` },
    { l: 'Correct', v: stats.correct, s: `POSITIVE ${stats.classes.POSITIVE.correct} · NEGATIVE ${stats.classes.NEGATIVE.correct}` },
    { l: 'Incorrect', v: stats.incorrect, s: `${stats.classes.POSITIVE.incorrect} in POSITIVE · ${stats.classes.NEGATIVE.incorrect} in NEGATIVE` },
    { l: 'Class mix', v: `${stats.classes.POSITIVE.count} / ${stats.classes.NEGATIVE.count}`, s: 'POSITIVE / NEGATIVE reviews' },
  ];
  $('#kpis').innerHTML = kpis.map(k =>
    `<div class="kpi"><p class="kpi-label">${k.l}</p><p class="kpi-value">${k.v}</p><p class="kpi-sub">${k.s}</p></div>`).join('');

  /* ---- Overall outcome bar ---- */
  $('#outcome').innerHTML =
    `<div class="legend"><span class="lg good">${stats.correct} correct (${stats.acc})</span>` +
    `<span class="lg bad">${stats.incorrect} incorrect (${stats.inc})</span></div>` +
    `<div class="hbar">${seg(stats.acc_num, 'good')}${seg(stats.inc_num, 'bad')}</div>`;

  /* ---- Outcome by class ---- */
  $('#byclass').innerHTML = ['POSITIVE', 'NEGATIVE'].map(c => {
    const s = stats.classes[c];
    return `<div class="cls"><div class="cls-head"><span class="cls-name">${c}</span>` +
      `<span class="cls-nums">${s.correct} correct · ${s.incorrect} incorrect (${s.acc})</span></div>` +
      `<div class="hbar">${seg(s.acc_num, 'good')}${seg(s.err_num, 'bad')}</div></div>`;
  }).join('');
  const p = stats.classes.POSITIVE, n = stats.classes.NEGATIVE;
  $('#byclass-note').textContent =
    `Mistakes by class: ${p.incorrect} of ${p.count} POSITIVE reviews (${p.err} error rate) and ` +
    `${n.incorrect} of ${n.count} NEGATIVE reviews (${n.err} error rate) — the rarer class is ` +
    `harder relative to its size.`;

  /* ---- Class mix donut ---- */
  const C = 2 * Math.PI * 48, posFrac = stats.classes.POSITIVE.count / stats.total;
  $('#mix').innerHTML =
    `<svg viewBox="0 0 120 120" width="142" height="142" role="img" aria-label="class mix donut">` +
      `<circle cx="60" cy="60" r="48" fill="none" stroke="#B5542F" stroke-width="18"/>` +
      `<circle cx="60" cy="60" r="48" fill="none" stroke="#2D7D5F" stroke-width="18" ` +
        `stroke-dasharray="${(posFrac * C).toFixed(1)} ${C.toFixed(1)}" transform="rotate(-90 60 60)"/>` +
      `<text x="60" y="61" text-anchor="middle" class="donut-num">${stats.total}</text>` +
      `<text x="60" y="76" text-anchor="middle" class="donut-cap">reviews</text>` +
    `</svg><div class="mix-legend">` +
      `<span class="lg pos">POSITIVE ${stats.classes.POSITIVE.count} (${(posFrac * 100).toFixed(1)}%)</span>` +
      `<span class="lg neg">NEGATIVE ${stats.classes.NEGATIVE.count} (${((1 - posFrac) * 100).toFixed(1)}%)</span>` +
    `</div>`;

  /* ---- Rating distribution ---- */
  const maxR = Math.max(...stats.ratings.map(x => x.count));
  $('#ratings').innerHTML = [...stats.ratings].reverse().map(x =>
    `<div class="rrow"><span class="rlab">${x.rating}★</span>` +
    `<span class="hbar small"><span class="seg gold" style="width:${(x.count / maxR * 100).toFixed(1)}%"></span></span>` +
    `<span class="rcount">${x.count}</span></div>`).join('');

  /* ---- Mismatch spotlight ---- */
  const bad = rows.filter(r => !r.correct);
  $('#mismatches').innerHTML =
    `<h2>Mismatched reviews <span class="count-badge">${bad.length}</span></h2>` +
    `<p class="note">Reviews where the model's label disagreed with the rating-derived label.` +
    ` Full text is shown for inspection.</p>` +
    bad.map(r => `<div class="mm"><div class="mm-head">` +
      `<span class="mm-idx">CSV index ${r.index}</span>` +
      `<span class="mm-rating" title="rating ${r.rating}">${stars(r.rating)}</span>` +
      `<span class="chip ${r.correct_label === 'POSITIVE' ? 'pos' : 'neg'}">expected ${r.correct_label}</span>` +
      `<span class="chip bad">predicted ${r.predicted_label}</span></div>` +
      `<p class="mm-title">${esc(r.title)}</p>` +
      `<p class="mm-text">${esc(r.text)}</p>` +
      (r.raw_output ? `<p class="mm-raw">model output: ${esc(r.raw_output)}</p>` : '') +
    `</div>`).join('');

  /* ---- Review table ---- */
  let filter = 'all', q = '', sortKey = 'index', sortDir = 1;
  const chipDefs = [['all', 'All', stats.total], ['correct', 'Correct', stats.correct],
    ['mismatch', 'Mismatch', stats.incorrect],
    ['POSITIVE', 'POSITIVE', stats.classes.POSITIVE.count],
    ['NEGATIVE', 'NEGATIVE', stats.classes.NEGATIVE.count]];
  $('#filters').innerHTML = chipDefs.map(([k, lab, n]) =>
    `<button type="button" class="fchip ${k === 'all' ? 'on' : ''}" data-f="${k}">${lab} <b>${n}</b></button>`).join('');
  $('#filters').addEventListener('click', e => {
    const b = e.target.closest('.fchip'); if (!b) return;
    filter = b.dataset.f;
    document.querySelectorAll('.fchip').forEach(x => x.classList.toggle('on', x === b));
    draw();
  });
  $('#search').addEventListener('input', e => { q = e.target.value.trim().toLowerCase(); draw(); });
  document.querySelector('#tbl thead').addEventListener('click', e => {
    const th = e.target.closest('th[data-key]'); if (!th) return;
    const k = th.dataset.key;
    if (sortKey === k) sortDir = -sortDir; else { sortKey = k; sortDir = k === 'index' ? 1 : -1; }
    document.querySelectorAll('th[data-key]').forEach(t => t.dataset.dir = (t === th) ? (sortDir > 0 ? 'asc' : 'desc') : '');
    draw();
  });
  const typeOf = k => k === 'index' || k === 'rating' ? 1 : k === 'correct' ? 2 : 0;
  const cmp = (a, b) => {
    const t = typeOf(sortKey), x = a[sortKey], y = b[sortKey];
    const d = t === 1 ? (x - y) : t === 2 ? (Number(x) - Number(y)) : String(x).localeCompare(String(y));
    return d * sortDir;
  };
  function visibleRows() {
    let a = rows.slice();
    if (filter === 'correct') a = a.filter(r => r.correct);
    else if (filter === 'mismatch') a = a.filter(r => !r.correct);
    else if (filter === 'POSITIVE' || filter === 'NEGATIVE') a = a.filter(r => r.correct_label === filter);
    if (q) a = a.filter(r => (r.title + ' ' + r.text + ' ' + r.asin).toLowerCase().includes(q));
    return a.sort(cmp);
  }
  function draw() {
    const vis = visibleRows();
    $('#tbody').innerHTML = vis.map(r =>
      `<tr class="${r.correct ? '' : 'mm-row'}">` +
      `<td class="num">${r.index}</td>` +
      `<td class="num" title="rating ${r.rating}">${stars(r.rating)}</td>` +
      `<td><span class="chip ${r.correct_label === 'POSITIVE' ? 'pos' : 'neg'}">${r.correct_label}</span></td>` +
      `<td><span class="chip ${r.predicted_label === 'POSITIVE' ? 'pos' : 'neg'} ${r.correct ? '' : 'warn'}">${r.predicted_label}</span></td>` +
      `<td><span class="chip ${r.correct ? 'good' : 'bad'}">${r.correct ? 'Correct' : 'Mismatch'}</span></td>` +
      `<td class="t-title">${esc(r.title)}</td>` +
      `<td class="t-text"><span class="t-short">${esc(shorten(r.text, 90))}</span>` +
      `<span class="t-full">${esc(r.text)}</span></td>` +
      `</tr>`).join('');
    $('#table-count').textContent = `${vis.length} of ${stats.total} rows`;
  }
  $('#tbody').addEventListener('click', e => {
    const tr = e.target.closest('tr'); if (!tr) return;
    tr.classList.toggle('open');
    const full = tr.querySelector('.t-full');
    if (full) full.hidden = !full.hidden;
  });
  draw();

  $('#foot').innerHTML =
    `Generated ${meta.generated} · read ${meta.total} rows from <b>${meta.source}</b> · ` +
    `model <b>${meta.model}</b> · endpoint <b>${meta.endpoint}</b> · ` +
    `self-contained, zero network requests · “index” is the CSV column (0-based)`;
  document.getElementById('chip-model').textContent = meta.model;
})();
</script>
</body>
</html>
"""


def main() -> None:
    if not os.path.isfile(CSV_PATH):
        raise SystemExit(f"{CSV_PATH} not found — run Step 2 first (python step2_classify.py).")

    rows = read_results(CSV_PATH)
    stats = compute_stats(rows)
    data = {
        "meta": {
            "source": CSV_PATH,
            "total": len(rows),
            "model": MODEL_NAME,
            "endpoint": ENDPOINT,
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "stats": stats,
        "rows": rows,
    }

    payload = json.dumps(data, ensure_ascii=False, indent=None).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", payload)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(html)

    # Internal self-check echoes the figures back so the run output is evidence.
    print(f"Dashboard written: {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e3:,.0f} KB)")
    print("Figures computed from step2_results.csv:")
    print(f"  total={stats['total']}  correct={stats['correct']}  incorrect={stats['incorrect']}"
          f"  accuracy={stats['acc']}")
    for c in ("POSITIVE", "NEGATIVE"):
        s = stats["classes"][c]
        print(f"  {c:<8} count={s['count']}  correct={s['correct']}  incorrect={s['incorrect']}"
              f"  class-acc={s['acc']}  error={s['err']}")
    rating_str = ", ".join(f"{x['rating']}★={x['count']}" for x in stats['ratings'])
    print(f"  rating counts: {rating_str}")
    print(f"  mismatch CSV indices: {stats['mismatch_indices']}")


if __name__ == "__main__":
    main()
