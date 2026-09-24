from ingestion.facts_store import FactStore, FinancialFact


def test_fact_store_save_and_query(tmp_path, monkeypatch):
    test_file = tmp_path / "test_facts.json"
    monkeypatch.setattr("ingestion.facts_store._FACTS_FILE", test_file)

    facts = [
        FinancialFact(
            ticker="NVDA",
            concept="Revenues",
            label="Total Revenue",
            fiscal_year=2025,
            quarter="FY",
            period_end="2025-01-26",
            value=60922.0,
            unit="USD",
            scale="millions",
        ),
        FinancialFact(
            ticker="NVDA",
            concept="OperatingIncomeLoss",
            label="Operating Income",
            fiscal_year=2025,
            quarter="FY",
            period_end="2025-01-26",
            value=32972.0,
            unit="USD",
            scale="millions",
        ),
        FinancialFact(
            ticker="WMT",
            concept="Revenues",
            label="Total Revenue",
            fiscal_year=2025,
            quarter="FY",
            period_end="2025-01-31",
            value=648125.0,
            unit="USD",
            scale="millions",
        ),
    ]

    FactStore.save(facts)
    loaded = FactStore.load(force_reload=True)
    assert len(loaded) == 3

    # Query by ticker & year
    res = FactStore.query(ticker="NVDA", fiscal_year=2025)
    assert len(res) == 2
    assert res[0].value == 60922.0

    # Query with concept filter
    res_op = FactStore.query(ticker="NVDA", fiscal_year=2025, concept="Operating")
    assert len(res_op) == 1
    assert res_op[0].label == "Operating Income"

    # Context formatting
    context = FactStore.format_as_context(res)
    assert "VERIFIED GAAP FINANCIAL FACTS" in context
    assert "NVDA" in context
    assert "$60,922.0 millions USD" in context


def test_fact_store_handles_none_quarter(tmp_path, monkeypatch):
    test_file = tmp_path / "test_facts_none_q.json"
    monkeypatch.setattr("ingestion.facts_store._FACTS_FILE", test_file)

    fact = FinancialFact(
        ticker="NFLX",
        concept="Revenues",
        label="Total Revenue",
        fiscal_year=2025,
        quarter=None,  # type: ignore[arg-type]
        period_end="2025-12-31",
        value=39000.0,
        unit="USD",
        scale="millions",
    )

    FactStore.save([fact])
    loaded = FactStore.load(force_reload=True)
    assert len(loaded) == 1

    res = FactStore.query(ticker="NFLX", fiscal_year=2025)
    assert len(res) == 1
    assert res[0].value == 39000.0
