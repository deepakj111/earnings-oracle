import os
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import settings as _settings

HEADERS = {
    "User-Agent": _settings.infra.sec_user_agent,
    "Accept-Encoding": "gzip, deflate",
}

COMPANIES = {
    "AAPL": "0000320193",
    "NVDA": "0001045810",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "META": "0001326801",
    "JPM": "0000019617",
    "XOM": "0000034088",
    "UNH": "0000072971",
    "TSLA": "0001318605",
    "WMT": "0000104169",
}


def get_8k_filings(
    cik: str,
    ticker: str,
    start_date: str = "2023-01-01",
    end_date: str = date.today().strftime("%Y-%m-%d"),
) -> list[dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    filings = data["filings"]["recent"]
    results = []
    for i, form in enumerate(filings["form"]):
        if form == "8-K":
            filing_date = filings["filingDate"][i]
            if start_date <= filing_date <= end_date:
                results.append(
                    {
                        "ticker": ticker,
                        "cik": cik,
                        "date": filing_date,
                        "accession": filings["accessionNumber"][i],
                    }
                )
    return results


def get_filing_documents(cik: str, accession: str) -> list[dict]:
    """
    Fetch the HTML filing index (the only guaranteed index format on EDGAR).
    URL: www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{accession}-index.htm
    Parse the document table to find all files and their types.
    """
    accession_clean = accession.replace("-", "")
    cik_int = int(cik)

    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{accession}-index.htm"
    )

    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"  Index failed ({resp.status_code}): {url}")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    docs = []

    # The index page has a table with columns: Seq, Description, Document, Type, Size
    for row in soup.select("table tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue
        description = cells[1].get_text(strip=True).lower()
        link_tag = cells[2].find("a")
        doc_name = link_tag.get_text(strip=True) if link_tag else cells[2].get_text(strip=True)
        doc_type = cells[3].get_text(strip=True)

        if doc_name:
            docs.append(
                {
                    "name": doc_name,
                    "type": doc_type,
                    "description": description,
                }
            )

    return docs


def pick_best_document(documents: list[dict]) -> str | None:
    # First pass: EX-99.1 / EX-99 is always the earnings press release
    for doc in documents:
        name = doc["name"].lower()
        doc_type = doc["type"]
        desc = doc["description"]

        if doc_type in ("EX-99.1", "EX-99.2", "EX-99"):
            return doc["name"]
        if "ex99" in name or "ex-99" in name:
            return doc["name"]
        if any(
            kw in desc for kw in ["earnings", "press release", "financial results", "exhibit 99"]
        ):
            return doc["name"]

    # Second pass: fallback to 8-K body
    for doc in documents:
        if doc["type"] == "8-K" and doc["name"]:
            return doc["name"]

    return None


def download_document(
    cik: str,
    accession: str,
    doc_name: str,
    filing_meta: dict,
    output_dir: str,
) -> str | None:
    accession_clean = accession.replace("-", "")
    cik_int = int(cik)

    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{doc_name}"
    resp = requests.get(url, headers=HEADERS, timeout=30)

    if resp.status_code != 200:
        print(f"  Download failed ({resp.status_code}): {url}")
        return None

    ticker = filing_meta["ticker"]
    filing_date = filing_meta["date"]
    safe_acc = accession_clean[:10]
    file_path = Path(output_dir) / f"{ticker}_{filing_date}_{safe_acc}.htm"
    file_path.write_text(resp.text, encoding="utf-8")
    print(f"  Downloaded: {file_path.name}  [{doc_name}]")
    return str(file_path)


if __name__ == "__main__":  # ← ADD THIS GUARD
    # --- Main ---
    os.makedirs("data/transcripts", exist_ok=True)

    all_filings = []
    for ticker, cik in COMPANIES.items():
        print(f"Fetching 8-K list for {ticker}...")
        filings = get_8k_filings(cik, ticker)
        all_filings.extend(filings)
        time.sleep(0.2)

    print(f"\nTotal 8-K filings found: {len(all_filings)}")
    print("Fetching document indexes and downloading exhibits...\n")

    success, skipped = 0, 0
    for filing in all_filings:
        documents = get_filing_documents(filing["cik"], filing["accession"])
        time.sleep(0.15)

        if not documents:
            skipped += 1
            continue

        best_doc = pick_best_document(documents)
        if not best_doc:
            print(f"  No exhibit found: {filing['ticker']} {filing['date']}")
            skipped += 1
            continue

        result = download_document(
            filing["cik"], filing["accession"], best_doc, filing, "data/transcripts"
        )
        success += 1 if result else 0
        skipped += 0 if result else 1
        time.sleep(0.15)

    print(f"\nDone: {success} downloaded, {skipped} skipped")
