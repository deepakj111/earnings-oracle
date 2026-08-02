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
    "NVDA": "0001045810",
    "JPM": "0000019617",
    "WMT": "0000104169",
    "TSLA": "0001318605",
}


def get_company_filings(
    cik: str,
    ticker: str,
    form_types: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
    start_date: str = "2020-01-01",
    end_date: str = date.today().strftime("%Y-%m-%d"),
) -> list[dict]:
    """Fetch recent 10-K, 10-Q, and 8-K filings for a specific company CIK within a date range."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    filings = data["filings"]["recent"]
    results = []
    for i, form in enumerate(filings["form"]):
        if form in form_types:
            filing_date = filings["filingDate"][i]
            if start_date <= filing_date <= end_date:
                primary_doc = (
                    filings["primaryDocument"][i] if "primaryDocument" in filings else None
                )
                results.append(
                    {
                        "ticker": ticker,
                        "cik": cik,
                        "form": form,
                        "date": filing_date,
                        "accession": filings["accessionNumber"][i],
                        "primary_doc": primary_doc,
                    }
                )
    return results


def get_8k_filings(
    cik: str,
    ticker: str,
    start_date: str = "2020-01-01",
    end_date: str = date.today().strftime("%Y-%m-%d"),
) -> list[dict]:
    """Fetch 8-K filings specifically for backward compatibility."""
    return get_company_filings(
        cik=cik,
        ticker=ticker,
        form_types=("8-K",),
        start_date=start_date,
        end_date=end_date,
    )


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


def pick_best_document(
    documents: list[dict],
    form_type: str = "8-K",
    primary_doc: str | None = None,
) -> str | None:
    """Select the most relevant report or exhibit document from a filing."""
    # For 10-K and 10-Q, prefer primary document or form matching document
    if form_type in ("10-K", "10-Q"):
        if primary_doc and any(d["name"] == primary_doc for d in documents):
            return primary_doc
        for doc in documents:
            if doc["type"] in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
                return doc["name"]

    # For 8-K: First pass check EX-99 press release exhibit
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

    # Fallback: check primary document or main body
    if primary_doc and any(d["name"] == primary_doc for d in documents):
        return primary_doc
    for doc in documents:
        if doc["type"] == form_type and doc["name"]:
            return doc["name"]

    return None


def download_document(
    cik: str,
    accession: str,
    doc_name: str,
    filing_meta: dict,
    output_dir: str,
) -> str | None:
    """Download the specific SEC EDGAR document HTML file."""
    accession_clean = accession.replace("-", "")
    cik_int = int(cik)

    url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{doc_name}"
    resp = requests.get(url, headers=HEADERS, timeout=30)

    if resp.status_code != 200:
        print(f"  Download failed ({resp.status_code}): {url}")
        return None

    ticker = filing_meta["ticker"]
    form = filing_meta.get("form", "8-K")
    filing_date = filing_meta["date"]
    safe_acc = accession_clean[:10]
    file_path = Path(output_dir) / f"{ticker}_{form}_{filing_date}_{safe_acc}.htm"
    file_path.write_text(resp.text, encoding="utf-8")
    print(f"  Downloaded: {file_path.name}  [{doc_name}]")
    return str(file_path)


def main() -> None:
    # --- Main ---
    os.makedirs("data/company_filings", exist_ok=True)

    all_filings = []
    for ticker, cik in COMPANIES.items():
        print(f"Fetching 10-K, 10-Q, 8-K filing lists for {ticker}...")
        filings = get_company_filings(cik, ticker)
        all_filings.extend(filings)
        time.sleep(0.2)

    print(f"\nTotal filings found across NVDA, JPM, WMT, TSLA: {len(all_filings)}")
    print("Fetching document indexes and downloading reports...\n")

    success, skipped = 0, 0
    for filing in all_filings:
        documents = get_filing_documents(filing["cik"], filing["accession"])
        time.sleep(0.15)

        if not documents:
            skipped += 1
            continue

        best_doc = pick_best_document(
            documents,
            form_type=filing.get("form", "8-K"),
            primary_doc=filing.get("primary_doc"),
        )
        if not best_doc:
            print(f"  No report found: {filing['ticker']} {filing['form']} {filing['date']}")
            skipped += 1
            continue

        result = download_document(
            filing["cik"], filing["accession"], best_doc, filing, "data/company_filings"
        )
        success += 1 if result else 0
        skipped += 0 if result else 1
        time.sleep(0.15)

    print(f"\nDone: {success} downloaded, {skipped} skipped")


if __name__ == "__main__":
    main()
