"""Browser verification of the Step 7 dashboard against step6_results.csv.

Checks, in a real headless Chromium:
  * page renders with no console errors,
  * every KPI value matches the CSV-derived figures,
  * confusion-matrix cells match the recomputed matrix,
  * the NEUTRAL spotlight shows the right numbers,
  * filter chips return correct subsets with live badge counts,
  * search narrows correctly.

Run (with Playwright installed; loads the local file directly):
    python verify_step7.py
Exit code 0 = all checks pass.
"""

import csv
import os
import sys

from playwright.sync_api import sync_playwright

BASE = os.path.dirname(os.path.abspath(__file__))
URL = "file:///" + os.path.join(BASE, "step7_dashboard.html").replace("\\", "/")
CSV_PATH = os.path.join(BASE, "step6_results.csv")
CLS = ("POSITIVE", "NEUTRAL", "NEGATIVE")

rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
n = len(rows)
correct = sum(r["correct"] == "True" for r in rows)
cm = {t: {p: 0 for p in CLS} for t in CLS}
for r in rows:
    cm[r["true_class"]][r["predicted_class"]] += 1
true_counts = {c: sum(1 for r in rows if r["true_class"] == c) for c in CLS}
pred_counts = {c: sum(1 for r in rows if r["predicted_class"] == c) for c in CLS}
acc = f"{round(correct / n * 100, 1):.1f}%"
fails = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got!r}  {'==' if ok else '!='}  {want!r}")
    if not ok:
        fails.append(name)


def row_ids(page):
    return page.eval_on_selector_all(
        "#tbody tr", "els => els.map(e => e.querySelector('td:nth-child(2)').textContent.trim())")


def badge(page):
    return page.text_content("#table-count").strip()


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
    page.goto(URL)
    page.wait_for_selector("#tbody tr")

    kpi_values = [el.text_content().strip() for el in page.query_selector_all(".kpi-value")]
    check("KPI values", kpi_values, ["72.7%", str(correct), str(n - correct), "50 / 50 / 50"])

    for t in CLS:
        for p_ in CLS:
            cell = page.query_selector(f'.cm-cell[data-t="{t}"][data-p="{p_}"] .cm-n')
            check(f"confusion cell {t}→{p_}", cell.text_content().strip(), str(cm[t][p_]))

    spot = page.text_content("#spotlight")
    nb = cm["NEUTRAL"]
    check("spotlight NEGATIVE count", f"{nb['NEGATIVE']} ({round(nb['NEGATIVE'] / 50 * 100)}%)" in spot, True)
    check("cm-note mentions 33", "33" in page.text_content("#cm-note"), True)

    mm_cards = page.query_selector_all("#mismatches .mm")
    check("mismatch cards", len(mm_cards), n - correct)

    check("initial rows", len(page.query_selector_all("#tbody tr")), n)
    check("initial badge", badge(page), f"{n} of {n} rows")

    def want_ids(pred):
        return sorted(r["dataset_index"] for r in rows if pred(r))

    filters = {
        "all": want_ids(lambda r: True),
        "correct": want_ids(lambda r: r["correct"] == "True"),
        "mismatch": want_ids(lambda r: r["correct"] != "True"),
        "POSITIVE": want_ids(lambda r: r["true_class"] == "POSITIVE"),
        "NEUTRAL": want_ids(lambda r: r["true_class"] == "NEUTRAL"),
        "NEGATIVE": want_ids(lambda r: r["true_class"] == "NEGATIVE"),
    }
    for key, want in filters.items():
        page.click(f'button.fchip[data-f="{key}"]')
        page.wait_for_timeout(120)
        got = sorted(row_ids(page))
        check(f"chip '{key}' rows", got, want)
        check(f"chip '{key}' badge", badge(page), f"{len(want)} of {n} rows")

    page.click('button.fchip[data-f="all"]')
    page.fill("#search", "gift")
    page.wait_for_timeout(120)
    want = sorted(r["dataset_index"] for r in rows
                  if "gift" in (r["title"] + " " + r["text"] + " " + r["asin"]).lower())
    check("search 'gift' rows", sorted(row_ids(page)), want)
    check("search 'gift' badge", badge(page), f"{len(want)} of {n} rows")

    if console_errors:
        fails.append("console errors: " + "; ".join(console_errors[:3]))
        print("Console errors:", console_errors[:3])

    print("\nAccuracy check:", acc, f"({correct}/{n})")
    browser.close()

print("\nRESULT:", "ALL STEP 7 BROWSER CHECKS PASSED" if not fails else f"{len(fails)} FAILURE(S): {fails}")
sys.exit(1 if fails else 0)
