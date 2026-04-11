import requests
import time
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SEC_USER_NAME = os.getenv("SEC_USER_NAME")
SEC_USER_EMAIL = os.getenv("SEC_USER_EMAIL")

if not SEC_USER_NAME or not SEC_USER_EMAIL:
    raise EnvironmentError("Missing SEC_USER_NAME or SEC_USER_EMAIL in .env file")

HEADERS = {
    "User-Agent": f"{SEC_USER_NAME} {SEC_USER_EMAIL}",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}

COMPANIES = {
    "AAPL": "0000320193",
    "NVDA": "0001045810",
    "MSFT": "0000789019",
    "AMZN": "0001018724",
    "META": "0001326801",
    "JPM":  "0000019617",
    "XOM":  "0000034088",
    "UNH":  "0000072971",
    "TSLA": "0001318605",
    "WMT":  "0000104169"
}

def get_8k_filings(cik: str, ticker: str, start_date="2023-01-01", end_date="2025-12-31"):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    filings = data["filings"]["recent"]
    results = []

    for i, form in enumerate(filings["form"]):
        if form == "8-K":
            date = filings["filingDate"][i]
            if start_date <= date <= end_date:
                results.append({
                    "ticker": ticker,
                    "cik": cik,
                    "date": date,
                    "accession": filings["accessionNumber"][i].replace("-", ""),
                    "primary_doc": filings["primaryDocument"][i]
                })

    return results

def download_transcript(filing: dict, output_dir: str):
    cik = filing["cik"]
    accession = filing["accession"]
    doc = filing["primary_doc"]

    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
    resp = requests.get(url, headers={**HEADERS, "Host": "www.sec.gov"})

    if resp.status_code == 200:
        file_path = Path(output_dir) / f"{filing['ticker']}_{filing['date']}_{accession[:10]}.htm"
        file_path.write_text(resp.text, encoding="utf-8")
        print(f"Downloaded: {file_path.name}")
        return str(file_path)
    else:
        print(f"Failed: {url} — Status {resp.status_code}")
        return None

if __name__ == "__main__":
    os.makedirs("data/transcripts", exist_ok=True)


    all_filings = []
    for ticker, cik in COMPANIES.items():
        print(f"Fetching 8-K list for {ticker}...")
        filings = get_8k_filings(cik, ticker)
        all_filings.extend(filings)
        time.sleep(0.2)

    print(f"\nTotal 8-K filings found: {len(all_filings)}")

    for filing in all_filings:
        download_transcript(filing, "data/transcripts")
        time.sleep(0.15)