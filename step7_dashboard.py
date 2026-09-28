"""MBAX 6418 - Assignment 1, Step 7: three-class dashboard for the Step 6
balanced results.

Reads step6_results.csv (supplemented by the identical rows saved in
step6_sample.csv) as the single source of truth -- no LLM calls -- and
generates a self-contained HTML dashboard in the same polished style as the
Step 3/4 dashboard (same design tokens, masthead, KPI cards, chips and table
mechanics), extended for three classes with:

  * star-rating distribution,
  * true class vs model-predicted class mix,
  * per-class accuracy (POSITIVE / NEUTRAL / NEGATIVE),
  * a heatmap confusion matrix with the NEUTRAL row spotlighted,
  * a dedicated "NEUTRAL problem" panel,
  * the review-level table with live filtering/search/counts.

Usage:
    python step7_dashboard.py          # writes step7_dashboard.html
"""

import csv
import datetime as _dt
import json
import os

from review_prompt import load_env_file

CSV_PATH_MAIN = "step6_results.csv"
CSV_PATH_SAMPLE = "step6_sample.csv"
OUT_PATH = "step7_dashboard.html"

CLASSES = ("POSITIVE", "NEUTRAL", "NEGATIVE")

# Provenance metadata (non-numeric): as configured for the Step 6 run.
load_env_file()
MODEL_NAME = os.environ.get("MODEL", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
ENDPOINT = os.environ.get("OPENAI_BASE_URL", "").rstrip("/") or "http://dobolyi.com:9001/v1"

# Full-dataset class shares, computed during the Step 6 run (152,410 reviews):
# POSITIVE 134,940 (88.5%), NEUTRAL 3,271 (2.1%), NEGATIVE 14,199 (9.3%).
FULL_DIST = {"POSITIVE": "88.5%", "NEUTRAL": "2.1%", "NEGATIVE": "9.3%"}


def pct(part: int, whole: int) -> str:
    return f"{round(100 * part / whole, 1):.1f}%" if whole else "n/a"


def pct_num(part: int, whole: int) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


def read_results() -> list[dict]:
    with open(CSV_PATH_MAIN, encoding="utf-8", newline="") as fh:
        raw = list(csv.DictReader(fh))
    rows = []
    for r in raw:
        rows.append({
            "n": len(rows) + 1,                     # 1-based sample ordinal
            "dataset_index": int(r["dataset_index"]),
            "asin": r["asin"],
            "title": r["title"],
            "text": r["text"],
            "rating": float(r["rating"]),
            "true_class": r["true_class"],
            "predicted_class": r["predicted_class"],
            "correct": r["correct"] == "True",
            "llm_raw": r["llm_raw"],
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
        "predicted": {c: 0 for c in CLASSES},
        "ratings": [],
        "confusion": {t: {p: 0 for p in CLASSES} for t in CLASSES},
        "neutral_breakdown": {},
        "mismatch_indices": [],
    }
    for label in CLASSES:
        sub = [r for r in rows if r["true_class"] == label]
        n, ok = len(sub), sum(r["correct"] for r in sub)
        stats["classes"][label] = {
            "count": n, "correct": ok, "incorrect": n - ok,
            "acc": pct(ok, n), "err": pct(n - ok, n),
            "acc_num": pct_num(ok, n), "err_num": pct_num(n - ok, n),
        }
    for r in rows:
        stats["predicted"][r["predicted_class"]] += 1
        stats["confusion"][r["true_class"]][r["predicted_class"]] += 1
        if not r["correct"]:
            stats["mismatch_indices"].append(r["dataset_index"])
    stats["neutral_breakdown"] = dict(stats["confusion"]["NEUTRAL"])
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
<title>MBAX 6418 · Assignment 1 · Three-Class Review Dashboard</title>
<style>
  :root{
    --paper:#F5F2EB; --card:#FFFFFF; --line:#E4DFD3; --ink:#1B1915; --mut:#6F6A5E;
    --gold:#C9A227; --gold-soft:#F3EAC8;
    --good:#1F6F52; --good-bg:#E5F1EB;
    --bad:#B4472C;  --bad-bg:#F9E9E3; --bad-hot:#8F3218; --bad-hot-bg:#F3DCD2;
    --pos:#2D7D5F;  --pos-bg:#E4F1EB;
    --neu:#8A6D0B;  --neu-bg:#F7F0D6;
    --neg:#B5542F;  --neg-bg:#F7E8E1;
    --serif:"Iowan Old Style","Palatino Linotype","Book Antiqua",Palatino,Georgia,serif;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --shadow:0 1px 2px rgba(27,25,21,.05),0 10px 28px -14px rgba(27,25,21,.16);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--paper);color:var(--ink);
       font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
  ::selection{background:var(--gold-soft)}

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
  .chip.pos{color:var(--pos);background:var(--pos-bg)}
  .chip.neu{color:var(--neu);background:var(--neu-bg)}
  .chip.neg{color:var(--neg);background:var(--neg-bg)}
  .chip.good{color:var(--good);background:var(--good-bg)}
  .chip.bad{color:var(--bad);background:var(--bad-bg)}
  .chip.warn{outline:1px dashed var(--bad);outline-offset:-1px}

  main{max-width:1180px;margin:0 auto;padding:26px 28px 8px}

  .kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:16px;margin-bottom:20px}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);
       padding:18px 20px 16px;position:relative;overflow:hidden}
  .kpi::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--gold)}
  .kpi-label{font:600 10.5px/1.2 var(--sans);letter-spacing:.16em;text-transform:uppercase;
             color:var(--mut);margin:0 0 8px}
  .kpi-value{font:600 34px/1.05 var(--serif);margin:0;letter-spacing:-.01em}
  .kpi-sub{margin:8px 0 0;font-size:12px;color:var(--mut)}

  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
        box-shadow:var(--shadow);padding:20px 22px}
  .card.full,.mismatches,.reviews{grid-column:1/-1}
  h2{font:600 11px/1 var(--sans);letter-spacing:.16em;text-transform:uppercase;color:var(--mut);
     margin:0 0 16px;display:flex;align-items:center;gap:9px}
  h2::before{content:"";width:14px;height:2px;background:var(--gold)}
  .count-badge{margin-left:auto;background:var(--ink);color:#fff;border-radius:999px;
               padding:3px 10px;font:600 11px/1.3 var(--sans)}
  .note{font-size:12.5px;color:var(--mut);margin:12px 0 0}

  .legend{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:10px}
  .lg{font:600 12.5px/1.3 var(--sans)}
  .lg::before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}
  .lg.good::before{background:var(--good)} .lg.bad::before{background:var(--bad)}
  .lg.pos::before{background:var(--pos)} .lg.neu::before{background:var(--neu)} .lg.neg::before{background:var(--neg)}
  .hbar{display:flex;height:34px;border-radius:9px;overflow:hidden;background:var(--line);gap:2px}
  .hbar .seg{display:block;height:100%}
  .hbar .seg.good{background:var(--good)} .hbar .seg.bad{background:var(--bad)}
  .hbar .seg.gold{background:var(--gold)} .hbar .seg.pos{background:var(--pos)}
  .hbar .seg.neu{background:var(--neu)} .hbar .seg.neg{background:var(--neg)}
  .hbar.small{height:12px;border-radius:6px;flex:1}

  .srow{margin-bottom:16px}
  .srow:last-child{margin-bottom:0}
  .shead{display:flex;justify-content:space-between;gap:10px;margin-bottom:7px;flex-wrap:wrap}
  .sname{font:700 13px/1.2 var(--sans);letter-spacing:.04em}
  .snums{font:600 12px/1.3 var(--sans);color:var(--mut)}

  .cls{margin-bottom:14px}
  .cls:last-child{margin-bottom:0}
  .cls-head{display:flex;justify-content:space-between;gap:10px;margin-bottom:7px;flex-wrap:wrap}
  .cls-name{font:700 13px/1.2 var(--sans);letter-spacing:.04em}
  .cls-name.neu{color:var(--neu)}
  .cls-nums{font:600 12px/1.3 var(--sans);color:var(--mut)}

  /* Confusion matrix */
  .cm-head,.cm-row{display:grid;grid-template-columns:150px repeat(3,1fr) 64px;gap:6px;margin-bottom:6px}
  .cm-head{font:600 10.5px/1.2 var(--sans);letter-spacing:.12em;text-transform:uppercase;color:var(--mut)}
  .cm-corner{color:var(--mut);padding-top:4px}
  .cm-label{font:700 12.5px/1.2 var(--sans);display:flex;align-items:center;padding-left:8px}
  .cm-label.neu{box-shadow:inset 3px 0 0 var(--gold);background:var(--gold-soft);border-radius:6px}
  .cm-cell{border-radius:8px;padding:9px 10px;text-align:center;min-height:56px;
           display:flex;flex-direction:column;justify-content:center}
  .cm-cell.ok{background:var(--good-bg);color:var(--good)}
  .cm-cell.bad{background:var(--bad-bg);color:var(--bad)}
  .cm-cell.hot{background:var(--bad-hot-bg);color:var(--bad-hot);box-shadow:inset 0 0 0 1px var(--bad)}
  .cm-n{font:700 21px/1 var(--serif)}
  .cm-p{font:600 10.5px/1 var(--sans);letter-spacing:.06em;margin-top:4px;opacity:.85}
  .cm-total{font:600 12px/1.3 var(--sans);color:var(--mut);display:flex;align-items:center;
            justify-content:center}
  .cm-total b{font-size:13px;color:var(--ink)}
  .cm-note{margin:12px 0 0;font-size:12.5px;color:var(--mut)}

  /* Ratings */
  .rrow{display:flex;align-items:center;gap:12px;margin-bottom:9px}
  .rlab{width:104px;color:#8A7F45;font-size:14px;letter-spacing:2px;white-space:nowrap}
  .rbar{flex:1;display:flex;height:14px;border-radius:7px;overflow:hidden;background:var(--line);gap:2px}
  .rcount{width:74px;text-align:right;font:600 12.5px/1 var(--sans);font-variant-numeric:tabular-nums}

  /* Mismatch cards */
  .mm{border-left:3px solid var(--bad);background:#FDF8F5;border-radius:10px;
      padding:12px 16px;margin-bottom:10px}
  .mm:last-child{margin-bottom:0}
  .mm-head{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:7px}
  .mm-idx{font:600 11px/1.2 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--mut)}
  .mm-rating{color:#8A7F45;font-size:13px;letter-spacing:2px}
  .mm-title{font:600 16px/1.35 var(--serif);margin:0 0 5px}
  .mm-text{margin:0;font-size:13.5px;color:#4A463D}

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
  .t-title{font-weight:600;min-width:200px;max-width:250px}
  .t-text{min-width:250px;max-width:400px;color:#4A463D;cursor:pointer}
  .t-text .t-full{color:var(--mut)}

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
      <p class="kicker">MBAX 6418 · Assignment 1 · Step 7</p>
      <h1>Review Classification Dashboard <span style="color:var(--gold)">— Three Classes</span></h1>
      <p class="sub">Amazon 2023 Gift Cards · balanced sample 50/50/50 (seed 42) ·
         3-class LLM labels vs rating-derived labels · ratings never shown to the model</p>
    </div>
    <div class="mh-meta">
      <span class="chip src">source: step6_results.csv</span>
      <span class="chip model" id="chip-model">model</span>
    </div>
  </div>
</header>

<main>
  <section class="kpis" id="kpis" aria-label="Key figures"></section>

  <section class="grid">
    <article class="card full">
      <h2>True class vs. model-predicted class</h2>
      <div id="truevs"></div>
    </article>

    <article class="card">
      <h2>Accuracy by true class</h2>
      <div id="byclass"></div>
      <p class="note">Each bar splits a class into correctly predicted (green) and wrong
         (red). NEUTRAL is the clear weak spot.</p>
    </article>

    <article class="card">
      <h2>The NEUTRAL problem</h2>
      <div id="spotlight"></div>
    </article>

    <article class="card full">
      <h2>Confusion matrix <span class="count-badge">rows = true · cols = predicted</span></h2>
      <div id="cm"></div>
      <p class="cm-note" id="cm-note"></p>
    </article>

    <article class="card full">
      <h2>Star-rating distribution of the sample</h2>
      <div id="ratings" style="max-width:640px"></div>
      <p class="note">Bar colors show which class each rating belongs to
         (1–2★ NEGATIVE · 3★ NEUTRAL · 4–5★ POSITIVE). For contrast, the full
         dataset is ${FULL_POS} POSITIVE / ${FULL_NEU} NEUTRAL / ${FULL_NEG} NEGATIVE —
         the sample deliberately balances a heavily positive dataset.</p>
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
          <th data-key="n" data-dir="asc">no.</th>
          <th data-key="dataset_index">dataset idx</th>
          <th data-key="rating">rating</th>
          <th data-key="true_class">true</th>
          <th data-key="predicted_class">predicted</th>
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
  const CLS = ['POSITIVE', 'NEUTRAL', 'NEGATIVE'];
  const $ = s => document.querySelector(s);
  const esc = s => String(s ?? '').replace(/[&<>"']/g,
    c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const stars = r => { const n = Math.max(1, Math.min(5, Math.round(r)));
    return '★'.repeat(n) + '<span style="opacity:.3">' + '☆'.repeat(5 - n) + '</span>'; };
  const seg = (w, c) => `<span class="seg ${c}" style="width:${w}%"></span>`;
  const shorten = (s, n) => s && s.length > n ? s.slice(0, n - 1) + '…' : s;
  const chipCls = c => c === 'POSITIVE' ? 'pos' : c === 'NEUTRAL' ? 'neu' : 'neg';

  /* ---- KPI cards ---- */
  const c = stats.classes;
  const kpis = [
    { l: 'Overall accuracy', v: stats.acc, s: `${stats.correct} of ${stats.total} reviews` },
    { l: 'Correct', v: stats.correct, s: `POSITIVE ${c.POSITIVE.correct} · NEUTRAL ${c.NEUTRAL.correct} · NEGATIVE ${c.NEGATIVE.correct}` },
    { l: 'Incorrect', v: stats.incorrect, s: `${c.POSITIVE.incorrect} POS · ${c.NEUTRAL.incorrect} NEU · ${c.NEGATIVE.incorrect} NEG` },
    { l: 'Class mix (true)', v: `${c.POSITIVE.count} / ${c.NEUTRAL.count} / ${c.NEGATIVE.count}`, s: 'POSITIVE / NEUTRAL / NEGATIVE · seed 42' },
  ];
  $('#kpis').innerHTML = kpis.map(k =>
    `<div class="kpi"><p class="kpi-label">${k.l}</p><p class="kpi-value">${k.v}</p><p class="kpi-sub">${k.s}</p></div>`).join('');

  /* ---- True vs predicted class mix ---- */
  function stacked(label, counts) {
    const tot = counts.POSITIVE + counts.NEUTRAL + counts.NEGATIVE;
    const p = x => `${x} (${(x / tot * 100).toFixed(1)}%)`;
    return `<div class="srow"><div class="shead"><span class="sname">${label}</span>
      <span class="snums">POSITIVE ${p(counts.POSITIVE)} · NEUTRAL ${p(counts.NEUTRAL)} · NEGATIVE ${p(counts.NEGATIVE)}</span></div>
      <div class="hbar">${seg(counts.POSITIVE / tot * 100, 'pos')}${seg(counts.NEUTRAL / tot * 100, 'neu')}${seg(counts.NEGATIVE / tot * 100, 'neg')}</div></div>`;
  }
  const trueCounts = { POSITIVE: c.POSITIVE.count, NEUTRAL: c.NEUTRAL.count, NEGATIVE: c.NEGATIVE.count };
  $('#truevs').innerHTML = stacked('True class', trueCounts) + stacked('Model predicted', stats.predicted);
  const extraNeg = stats.predicted.NEGATIVE - trueCounts.NEGATIVE;
  $('#truevs').insertAdjacentHTML('beforeend',
    `<p class="note">The model predicts NEGATIVE ${stats.predicted.NEGATIVE}× but only ${trueCounts.NEGATIVE} reviews are truly negative (${extraNeg > 0 ? '+' + extraNeg + ' extra' : extraNeg}), and it predicts NEUTRAL only ${stats.predicted.NEUTRAL}× despite ${trueCounts.NEUTRAL} true neutrals. Its vocabulary runs to the extremes.</p>`);

  /* ---- Accuracy by true class ---- */
  $('#byclass').innerHTML = CLS.map(k => {
    const s = c[k];
    return `<div class="cls"><div class="cls-head"><span class="cls-name${k === 'NEUTRAL' ? ' neu' : ''}">${k}</span>` +
      `<span class="cls-nums">${s.correct} correct · ${s.incorrect} wrong (${s.acc})</span></div>` +
      `<div class="hbar">${seg(s.acc_num, 'good')}${seg(s.err_num, 'bad')}</div></div>`;
  }).join('');

  /* ---- NEUTRAL spotlight ---- */
  const nb = stats.neutral_breakdown, totNeu = c.NEUTRAL.count;
  $('#spotlight').innerHTML =
    `<div class="shead" style="margin-bottom:7px"><span class="sname">3★ NEUTRAL reviews</span>` +
    `<span class="snums">${c.NEUTRAL.acc} correctly predicted</span></div>` +
    `<div class="hbar">${seg(nb.POSITIVE / totNeu * 100, 'pos')}${seg(nb.NEUTRAL / totNeu * 100, 'neu')}${seg(nb.NEGATIVE / totNeu * 100, 'neg')}</div>` +
    `<p class="note">Of ${totNeu} three-star reviews, ${nb.NEUTRAL} (${(nb.NEUTRAL / totNeu * 100).toFixed(0)}%) were correctly called NEUTRAL, ` +
    `but ${nb.NEGATIVE} (${(nb.NEGATIVE / totNeu * 100).toFixed(0)}%) were called NEGATIVE and ${nb.POSITIVE} ` +
    `(${(nb.POSITIVE / totNeu * 100).toFixed(0)}%) POSITIVE. Many 3★ reviews read as disappointment, so the model pushes them into NEGATIVE.</p>`;

  /* ---- Confusion matrix ---- */
  const cmHead = `<div class="cm-head"><span class="cm-corner"></span>` +
    CLS.map(k => `<span>${k}</span>`).join('') + `<span></span></div>`;
  const cmRows = CLS.map(t => {
    const cells = CLS.map(p => {
      const v = stats.confusion[t][p];
      const okCell = t === p;
      const hot = !okCell && v >= 30;
      const pctRow = `${(v / Math.max(1, c[t].count) * 100).toFixed(0)}%`;
      return `<div class="cm-cell ${okCell ? 'ok' : hot ? 'hot' : 'bad'}" data-t="${t}" data-p="${p}" ` +
        `title="true ${t} → predicted ${p}: ${v} of ${c[t].count}">` +
        `<span class="cm-n">${v}</span><span class="cm-p">${pctRow} of ${c[t].count}</span></div>`;
    }).join('');
    return `<div class="cm-row"><span class="cm-label${t === 'NEUTRAL' ? ' neu' : ''}">${t}</span>` +
      cells + `<span class="cm-total"><b>${c[t].count}</b></span></div>`;
  }).join('');
  $('#cm').innerHTML = cmHead + cmRows +
    `<div class="legend" style="margin-top:12px">
       <span class="lg good">correct</span><span class="lg bad">wrong</span><span class="lg bad" style="filter:brightness(.85)">≥ 30 wrong</span>
     </div>`;
  $('#cm-note').textContent =
    `Biggest error cell: ${nb.NEGATIVE} NEUTRAL reviews predicted NEGATIVE (${(nb.NEGATIVE / totNeu * 100).toFixed(0)}% of the 3★ class). ` +
    `The model is nearly perfect on the extremes (${c.POSITIVE.acc} POSITIVE, ${c.NEGATIVE.acc} NEGATIVE) and fails on the middle.`;

  /* ---- Star rating distribution ---- */
  const maxR = Math.max(...stats.ratings.map(x => x.count));
  $('#ratings').innerHTML = [...stats.ratings].reverse().map(x => {
    const cls = x.rating <= 2 ? 'neg' : x.rating === 3 ? 'neu' : 'pos';
    return `<div class="rrow"><span class="rlab">${x.rating}★</span>` +
      `<span class="rbar"><span class="seg ${cls}" style="width:${(x.count / maxR * 100).toFixed(1)}%"></span></span>` +
      `<span class="rcount">${x.count} · ${(x.count / stats.total * 100).toFixed(1)}%</span></div>`;
  }).join('');

  /* ---- Mismatch cards ---- */
  const bad = rows.filter(r => !r.correct);
  $('#mismatches').innerHTML =
    `<h2>Mismatched reviews <span class="count-badge">${bad.length}</span></h2>` +
    `<p class="note">All reviews where the model's 3-class label disagreed with the rating-derived label, for inspection.</p>` +
    bad.map(r => `<div class="mm"><div class="mm-head">` +
      `<span class="mm-idx">dataset idx ${r.dataset_index}</span>` +
      `<span class="mm-rating" title="rating ${r.rating}">${stars(r.rating)}</span>` +
      `<span class="chip ${chipCls(r.true_class)}">true ${r.true_class}</span>` +
      `<span class="chip bad">predicted ${r.predicted_class}</span></div>` +
      `<p class="mm-title">${esc(r.title)}</p>` +
      `<p class="mm-text">${esc(r.text)}</p></div>`).join('');

  /* ---- Review table ---- */
  let filter = 'all', q = '', sortKey = 'n', sortDir = 1;
  const chipDefs = [['all', 'All', stats.total], ['correct', 'Correct', stats.correct],
    ['mismatch', 'Mismatch', stats.incorrect],
    ['POSITIVE', 'POSITIVE', c.POSITIVE.count],
    ['NEUTRAL', 'NEUTRAL', c.NEUTRAL.count],
    ['NEGATIVE', 'NEGATIVE', c.NEGATIVE.count]];
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
    if (sortKey === k) sortDir = -sortDir; else { sortKey = k; sortDir = k === 'n' ? 1 : -1; }
    document.querySelectorAll('th[data-key]').forEach(t => t.dataset.dir = (t === th) ? (sortDir > 0 ? 'asc' : 'desc') : '');
    draw();
  });
  const typeOf = k => ['n', 'dataset_index', 'rating'].includes(k) ? 1 : k === 'correct' ? 2 : 0;
  const cmp = (a, b) => {
    const t = typeOf(sortKey), x = a[sortKey], y = b[sortKey];
    const d = t === 1 ? (x - y) : t === 2 ? (Number(x) - Number(y)) : String(x).localeCompare(String(y));
    return d * sortDir;
  };
  function visibleRows() {
    let a = rows.slice();
    if (filter === 'correct') a = a.filter(r => r.correct);
    else if (filter === 'mismatch') a = a.filter(r => !r.correct);
    else if (CLS.includes(filter)) a = a.filter(r => r.true_class === filter);
    if (q) a = a.filter(r => (r.title + ' ' + r.text + ' ' + r.asin).toLowerCase().includes(q));
    return a.sort(cmp);
  }
  function draw() {
    const vis = visibleRows();
    $('#tbody').innerHTML = vis.map(r =>
      `<tr class="${r.correct ? '' : 'mm-row'}">` +
      `<td class="num">${r.n}</td>` +
      `<td class="num">${r.dataset_index}</td>` +
      `<td class="num" title="rating ${r.rating}">${stars(r.rating)}</td>` +
      `<td><span class="chip ${chipCls(r.true_class)}">${r.true_class}</span></td>` +
      `<td><span class="chip ${chipCls(r.predicted_class)} ${r.correct ? '' : 'warn'}">${r.predicted_class}</span></td>` +
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
    `Generated ${meta.generated} · read ${meta.total} rows from <b>${meta.source}</b> ` +
    `(sample <b>${meta.sample}</b>) · model <b>${meta.model}</b> · endpoint <b>${meta.endpoint}</b> · ` +
    `self-contained, zero network requests · “dataset idx” is the line number in Gift_Cards.jsonl.gz`;
  document.getElementById('chip-model').textContent = meta.model;
})();
</script>
</body>
</html>
"""


def main() -> None:
    if not os.path.isfile(CSV_PATH_MAIN):
        raise SystemExit(f"{CSV_PATH_MAIN} not found — run Step 6 first "
                         f"(python step6_balanced.py --k 50 --seed 42).")

    rows = read_results()
    stats = compute_stats(rows)
    data = {
        "meta": {
            "source": CSV_PATH_MAIN,
            "sample": f"{CSV_PATH_SAMPLE} (k=50, seed 42)",
            "total": len(rows),
            "model": MODEL_NAME,
            "endpoint": ENDPOINT,
            "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "stats": stats,
        "rows": rows,
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    # fill the full-dataset contrast numbers into the ratings footnote
    html = (html.replace("${FULL_POS}", FULL_DIST["POSITIVE"])
                .replace("${FULL_NEU}", FULL_DIST["NEUTRAL"])
                .replace("${FULL_NEG}", FULL_DIST["NEGATIVE"]))
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"Dashboard written: {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e3:,.0f} KB)")
    print("Figures computed from step6_results.csv:")
    print(f"  total={stats['total']}  correct={stats['correct']}  incorrect={stats['incorrect']}"
          f"  accuracy={stats['acc']}")
    for k in CLASSES:
        s = stats["classes"][k]
        print(f"  {k:<8} count={s['count']}  correct={s['correct']}  incorrect={s['incorrect']}"
              f"  acc={s['acc']}  err={s['err']}")
    print(f"  predicted counts: {stats['predicted']}")
    print(f"  confusion: {stats['confusion']}")
    print(f"  neutral breakdown: {stats['neutral_breakdown']}")
    print(f"  rating counts: "
          f"{', '.join(str(x['count']) for x in stats['ratings'])} (1★..5★)")
    print(f"  mismatches: {len(stats['mismatch_indices'])}")


if __name__ == "__main__":
    main()
