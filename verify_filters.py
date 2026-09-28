"""Verify the dashboard's interactive filtering (Step 4 readiness check).

Loads step3_dashboard.html in a headless Chromium and, for every filter
chip plus search, asserts that the rendered rows and the live count badge
match ground truth recomputed from step2_results.csv. Sorting and a
filter+search combination are checked too.

Needs the page served over http (run `python -m http.server 8791` in this
folder) and Playwright installed:
    python -m pip install playwright && python -m playwright install chromium

Usage:
    python verify_filters.py        # exit code 0 = all checks pass
"""

import csv
import sys

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8791/step3_dashboard.html"

csv_rows = list(csv.DictReader(open("step2_results.csv", encoding="utf-8")))


def ids(pred):
    return sorted(int(r["index"]) for r in csv_rows if pred(r))


def search_ids(term, extra=None):
    return sorted(int(r["index"]) for r in csv_rows
                  if (extra is None or extra(r))
                  and term in (r["title"] + " " + r["text"] + " " + r["asin"]).lower())


expected = {
    "all": ids(lambda r: True),
    "correct": ids(lambda r: r["correct"] == "True"),
    "mismatch": ids(lambda r: r["correct"] != "True"),
    "POSITIVE": ids(lambda r: r["correct_label"] == "POSITIVE"),
    "NEGATIVE": ids(lambda r: r["correct_label"] == "NEGATIVE"),
}

fails = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got if len(str(got)) < 80 else str(got)[:60] + '…'}"
          f"{'  ==  ' + str(want)[:60] + '…' if len(str(want)) > 60 else '  ==  ' + str(want) if ok else '  !=  ' + str(want)}")
    if not ok:
        fails.append(name)


def row_ids(page):
    return page.eval_on_selector_all(
        "#tbody tr", "els => els.map(e => parseInt(e.querySelector('td').textContent))")


def badge_count(page):
    return int(page.text_content("#table-count").strip().split()[0])


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    console_errors = []
    page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
    page.goto(URL)
    page.wait_for_selector("#tbody tr")
    print(f"Loaded {URL}; initial row count = {len(row_ids(page))}\n")

    # 1) each filter chip: rendered row indices + live badge count
    for key, want in expected.items():
        page.click(f'button.fchip[data-f="{key}"]')
        page.wait_for_timeout(120)
        check(f"chip '{key}': rendered rows", row_ids(page), want)
        check(f"chip '{key}': live badge", badge_count(page), len(want))
        if key in ("correct", "mismatch"):
            statuses = sorted(set(page.eval_on_selector_all(
                "#tbody tr td:nth-child(5)", "els => els.map(e => e.textContent.trim())")))
            check(f"chip '{key}': status column", statuses, ["Correct" if key == "correct" else "Mismatch"])

    # 2) search
    page.click('button.fchip[data-f="all"]')
    page.fill("#search", "note")
    page.wait_for_timeout(120)
    check("search 'note': rendered rows", row_ids(page), search_ids("note"))
    check("search 'note': live badge", badge_count(page), len(search_ids("note")))

    # 3) combined filter + search
    page.fill("#search", "")
    page.click('button.fchip[data-f="POSITIVE"]')
    page.fill("#search", "easy")
    page.wait_for_timeout(120)
    want = search_ids("easy", lambda r: r["correct_label"] == "POSITIVE")
    check("POSITIVE + search 'easy': rendered rows", row_ids(page), want)
    check("POSITIVE + search 'easy': live badge", badge_count(page), len(want))

    # 4) sorting: rating desc -> first row is index 0 (a 5.0); asc -> index 4 (first 1.0)
    page.fill("#search", "")
    page.click('button.fchip[data-f="all"]')
    page.click('th[data-key="rating"]')
    page.wait_for_timeout(120)
    first_desc = page.text_content("#tbody tr:first-child td:nth-child(1)").strip()
    page.click('th[data-key="rating"]')
    page.wait_for_timeout(120)
    first_asc = page.text_content("#tbody tr:first-child td:nth-child(1)").strip()
    check("sort rating desc first row", first_desc, "0")
    check("sort rating asc first row", first_asc, "4")

    if console_errors:
        fails.append("console errors: " + "; ".join(console_errors[:3]))
        print("Console errors:", console_errors[:3])
    browser.close()

print("\nRESULT:", "ALL FILTER CHECKS PASSED" if not fails else f"{len(fails)} FAILURE(S): {fails}")
sys.exit(1 if fails else 0)
