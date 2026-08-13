import os
import time
from datetime import date
from pathlib import Path
from typing import TypedDict

import requests
from bs4 import BeautifulSoup

from config import settings as _settings


class FormSummary(TypedDict):
    count: int
    years: set[str]


HEADERS = {
    "User-Agent": _settings.infra.sec_user_agent,
    "Accept-Encoding": "gzip, deflate",
}

# Added diverse popular companies across different sectors
COMPANIES = {
    "NVDA": "0001045810",  # Technology / Semiconductors
    "WMT": "0000104169",  # Consumer Discretionary / Retail
    "UNH": "0000731766",  # Healthcare / Managed Care
    "NFLX": "0001065280",  # Comm Services / Streaming (Concise business model)
}


def get_company_filings(
    cik: str,
    ticker: str,
    form_types: tuple[str, ...] = ("10-K",),
    start_date: str = "2025-01-01",
    end_date: str = date.today().strftime("%Y-%m-%d"),
) -> list[dict]:
    """Fetch 10-K filings for a specific company CIK within a date range, including older files."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Helper function to extract relevant filings from a dictionary of lists
    def extract_filings(filing_dict: dict) -> list[dict]:
        results = []
        for i, form in enumerate(filing_dict["form"]):
            if form in form_types:
                filing_date = filing_dict["filingDate"][i]
                if start_date <= filing_date <= end_date:
                    primary_doc = (
                        filing_dict["primaryDocument"][i]
                        if "primaryDocument" in filing_dict
                        else None
                    )
                    results.append(
                        {
                            "ticker": ticker,
                            "cik": cik,
                            "form": form,
                            "date": filing_date,
                            "accession": filing_dict["accessionNumber"][i],
                            "primary_doc": primary_doc,
                        }
                    )
        return results

    # 1. Get recent filings (usually the last ~1000 filings)
    all_results = extract_filings(data["filings"]["recent"])

    # 2. Fetch older filing archives if they overlap with our start_date
    if "files" in data["filings"]:
        for archive in data["filings"]["files"]:
            # If the archive's latest filing (filingTo) is >= our start_date, it might contain what we need
            if archive["filingTo"] >= start_date:
                archive_url = f"https://data.sec.gov/submissions/{archive['name']}"
                archive_resp = requests.get(archive_url, headers=HEADERS, timeout=30)
                if archive_resp.status_code == 200:
                    archive_data = archive_resp.json()
                    all_results.extend(extract_filings(archive_data))
                time.sleep(0.15)  # SEC rate limit padding

    return all_results


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
    form_type: str = "10-K",
    primary_doc: str | None = None,
) -> str | None:
    """Select the most relevant report document from a filing."""
    # Prefer primary document if present
    if primary_doc and any(d["name"] == primary_doc for d in documents):
        return primary_doc

    # Search for matching document type
    for doc in documents:
        if doc["type"] in (form_type, f"{form_type}/A"):
            return doc["name"]

    # Fallback to any document matching form_type
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
    form = filing_meta.get("form", "10-K")
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
        print(f"Fetching 10-K, 10-Q filing lists for {ticker}...")
        filings = get_company_filings(
            cik, ticker, form_types=("10-K", "10-Q"), start_date="2025-01-01"
        )
        all_filings.extend(filings)
        time.sleep(0.15)

    print(f"\nTotal filings found across {', '.join(COMPANIES.keys())}: {len(all_filings)}")
    print("Fetching document indexes and downloading reports...\n")

    # Data structure to hold summary info
    # ticker -> form_type -> {"count": int, "years": set}
    summary: dict[str, dict[str, FormSummary]] = {
        ticker: {"10-K": {"count": 0, "years": set()}, "10-Q": {"count": 0, "years": set()}}
        for ticker in COMPANIES
    }

    # Tracking list to hold (file_name, size_in_bytes)
    downloaded_files_info: list[tuple[str, int]] = []

    success, skipped = 0, 0
    for filing in all_filings:
        ticker = filing["ticker"]
        form = filing["form"]
        year = filing["date"][:4]  # Extract the year from YYYY-MM-DD

        documents = get_filing_documents(filing["cik"], filing["accession"])
        time.sleep(0.15)

        if not documents:
            skipped += 1
            continue

        best_doc = pick_best_document(
            documents,
            form_type=form,
            primary_doc=filing.get("primary_doc"),
        )
        if not best_doc:
            print(f"  No report found: {ticker} {form} {filing['date']}")
            skipped += 1
            continue

        result = download_document(
            filing["cik"], filing["accession"], best_doc, filing, "data/company_filings"
        )

        if result:
            success += 1
            summary[ticker][form]["count"] += 1
            summary[ticker][form]["years"].add(year)

            # Record the file name and size
            file_size = os.path.getsize(result)
            downloaded_files_info.append((Path(result).name, file_size))
        else:
            skipped += 1

        time.sleep(0.15)

    print(f"\nDone: {success} downloaded, {skipped} skipped")

    # --- Print Summary ---
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY".center(50))
    print("=" * 50)
    for ticker, forms in summary.items():
        print(f"\n{ticker}:")
        for form_type, data in forms.items():
            count = data["count"]
            years = sorted(data["years"])
            if count > 0:
                years_str = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
                print(f"  - {form_type}: {count} files downloaded (Covering: {years_str})")
            else:
                print(f"  - {form_type}: 0 files downloaded")

    # --- Print Sorted File Sizes ---
    if downloaded_files_info:
        # Sort descending by file size (index 1 is the size in bytes)
        downloaded_files_info.sort(key=lambda x: x[1], reverse=True)

        print("\n" + "=" * 50)
        print("DOWNLOADED FILES BY SIZE (Descending)".center(50))
        print("=" * 50)
        for file_name, file_size in downloaded_files_info:
            size_mb = file_size / (1024 * 1024)
            print(f"{size_mb:>6.2f} MB | {file_name}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
