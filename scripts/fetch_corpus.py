"""Fetch the latest 10-K for each ticker from SEC EDGAR and clean it to text.

Uses the EDGAR submissions API (data.sec.gov) to find the most recent 10-K,
downloads the primary document from the Archives, and converts the HTML to
plain text that preserves Item / section headings (Item 1, Item 1A, ...)
as standalone lines so structure-aware chunking can split on them.

EDGAR fair-access rules: a declared User-Agent is required and requests
must stay under 10/sec; this script sleeps between requests.

Usage:  python scripts/fetch_corpus.py [--out data/corpus]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

USER_AGENT = "Rohith Sunkishala research sunkishalarohith853@gmail.com"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# ticker -> CIK (zero-padded to 10 digits for the submissions API)
TICKERS = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
}

SLEEP_SECONDS = 0.6


def get_json(url: str) -> dict:
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    time.sleep(SLEEP_SECONDS)
    return resp.json()


def latest_10k(cik: str) -> dict:
    """Return {accession, primary_document, filing_date} for the newest 10-K."""
    data = get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    recent = data["filings"]["recent"]
    best = None
    for form, date, accession, doc in zip(
        recent["form"],
        recent["filingDate"],
        recent["accessionNumber"],
        recent["primaryDocument"],
    ):
        if form == "10-K" and (best is None or date > best["filing_date"]):
            best = {
                "accession": accession,
                "primary_document": doc,
                "filing_date": date,
            }
    if best is None:
        raise RuntimeError(f"No 10-K found for CIK {cik}")
    return best


def html_to_text(html: str) -> str:
    """Convert filing HTML to plain text, keeping headings on their own lines."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Block-level breaks become newlines so headings stay standalone lines.
    text = soup.get_text("\n")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[\s\u00a0]+", " ", line).strip()
        if line:
            lines.append(line)
    # Table-of-contents entries often split "Item 1." and its title onto
    # separate lines; rejoin them so headings carry their titles.
    merged: list[str] = []
    for line in lines:
        if (
            merged
            and re.fullmatch(r"Item\s+\d+[A-Z]?\.", merged[-1])
            and len(line) < 60
            and not line[0].isdigit()
        ):
            merged[-1] = f"{merged[-1]} {line}"
        else:
            merged.append(line)
    text = "\n".join(merged)
    # A heading that got glued to the following text gets its own line back.
    text = re.sub(r"(?<!\n)(Item\s+\d+[A-Z]?\.)", r"\n\1", text)
    return text.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/corpus")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for ticker, cik in TICKERS.items():
        filing = latest_10k(cik)
        accession_nodash = filing["accession"].replace("-", "")
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession_nodash}/{filing['primary_document']}"
        )
        print(f"{ticker}: 10-K filed {filing['filing_date']} -> {url}")
        resp = requests.get(url, headers=HEADERS, timeout=300)
        resp.raise_for_status()
        time.sleep(SLEEP_SECONDS)
        text = html_to_text(resp.text)
        path = out_dir / f"{ticker}_10k.txt"
        path.write_text(text, encoding="utf-8")
        words = len(text.split())
        print(f"  wrote {path}  ({len(text):,} chars, {words:,} words)")
        manifest[ticker] = {**filing, "url": url, "chars": len(text), "words": words}

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
