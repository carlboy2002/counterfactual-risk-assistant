"""
edgar_utils.py
--------------
Handles all SEC EDGAR data acquisition and HTML/text preprocessing.
Fetches the full 10-K filing text (not just Item 1A) for a given
ticker and fiscal year, then cleans it for downstream chunking.
"""

import re
import time
import logging
import requests
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# SEC EDGAR requires a descriptive User-Agent per their policy:
# https://www.sec.gov/os/accessing-edgar-data
# !! IMPORTANT: Replace with your real name and email below !!
# ----------------------------------------------------------------
USER_AGENT = "Columbia University wx2344@columbia.edu"


def _www_headers():
    """Headers for requests to www.sec.gov"""
    return {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }

def _data_headers():
    """Headers for requests to data.sec.gov (no Host override)"""
    return {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


EDGAR_FILING_BASE = "https://www.sec.gov/Archives/edgar/data"


def get_cik_for_ticker(ticker: str) -> Optional[int]:
    """
    Resolve a stock ticker to its SEC Central Index Key (CIK).
    Uses the SEC's canonical company_tickers.json file.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=_www_headers(), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            cik = int(entry["cik_str"])
            logger.info(f"Resolved {ticker_upper} -> CIK {cik}")
            return cik
    return None


def _search_filings(forms, dates, accession_numbers, primary_docs, cik, year):
    """Scan a filing list for the target 10-K year."""
    for i, form in enumerate(forms):
        if form not in ("10-K", "10-K/A"):
            continue
        filing_year = int(dates[i].split("-")[0])
        if filing_year == year or filing_year == year + 1:
            acc = accession_numbers[i].replace("-", "")
            doc_url = f"{EDGAR_FILING_BASE}/{cik}/{acc}/{primary_docs[i]}"
            logger.info(f"Found 10-K: {doc_url} (filed {dates[i]})")
            return doc_url
    return None


def get_10k_filing_url(cik: int, year: int) -> Optional[str]:
    """
    Find the primary 10-K document URL for a given CIK and fiscal year.
    """
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    logger.info(f"Fetching submissions: {submissions_url}")

    resp = requests.get(submissions_url, headers=_data_headers(), timeout=20)
    resp.raise_for_status()
    data = resp.json()

    filings = data.get("filings", {}).get("recent", {})
    result = _search_filings(
        filings.get("form", []),
        filings.get("filingDate", []),
        filings.get("accessionNumber", []),
        filings.get("primaryDocument", []),
        cik, year,
    )
    if result:
        return result

    # Check older archived filing pages
    for file_info in data.get("filings", {}).get("files", []):
        try:
            older_url = f"https://data.sec.gov/submissions/{file_info['name']}"
            older_resp = requests.get(older_url, headers=_data_headers(), timeout=15)
            if older_resp.ok:
                od = older_resp.json()
                result = _search_filings(
                    od.get("form", []), od.get("filingDate", []),
                    od.get("accessionNumber", []), od.get("primaryDocument", []),
                    cik, year,
                )
                if result:
                    return result
        except Exception as e:
            logger.warning(f"Older filings page error: {e}")

    return None


def fetch_and_clean_10k(ticker: str, year: int) -> tuple:
    """
    Main entry point: fetch and clean a 10-K filing.
    Returns: (clean_text, source_url)
    """
    logger.info(f"Fetching 10-K for {ticker.upper()} ({year})")

    cik = get_cik_for_ticker(ticker)
    if not cik:
        raise ValueError(f"Could not resolve CIK for ticker '{ticker}'.")

    filing_url = get_10k_filing_url(cik, year)
    if not filing_url:
        raise ValueError(
            f"No 10-K filing found for {ticker.upper()} FY{year}. "
            "Try an adjacent year or verify the ticker."
        )

    time.sleep(0.3)
    resp = requests.get(filing_url, headers=_www_headers(), timeout=30)
    resp.raise_for_status()

    clean_text = _parse_and_clean_html(resp.text)

    if len(clean_text) < 1000:
        raise ValueError(f"Extracted text too short ({len(clean_text)} chars).")

    logger.info(f"Clean text: {len(clean_text):,} characters")
    return clean_text, filing_url


def _parse_and_clean_html(raw_html: str) -> str:
    """Strip HTML and normalize whitespace from a 10-K document."""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "meta", "link", "head"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) > 40]
    text = "\n\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"Table of Contents\s*", "", text, flags=re.IGNORECASE)
    return text.strip()
