from generation.calculator import SafeFinancialCalculator


def test_basic_arithmetic():
    calc = SafeFinancialCalculator()
    assert calc.evaluate("2 + 2").result == 4.0
    assert calc.evaluate("100 * (1 - 0.25)").result == 75.0
    assert calc.evaluate("10 / 4").result == 2.5
    assert calc.evaluate("10 // 3").result == 3.0
    assert calc.evaluate("10 % 3").result == 1.0


def test_financial_helpers():
    calc = SafeFinancialCalculator()
    # growth: (new - old) / old * 100
    res = calc.evaluate("growth(100, 150)")
    assert res.success is True
    assert res.result == 50.0

    # margin: num / denom * 100
    res = calc.evaluate("margin(25, 100)")
    assert res.result == 25.0

    # bps: (new_pct - old_pct) * 100
    res = calc.evaluate("bps(15.5, 17.0)")
    assert res.result == 150.0

    # round
    res = calc.evaluate("round(margin(33, 100), 1)")
    assert res.result == 33.0


def test_security_sandboxing():
    calc = SafeFinancialCalculator()
    # No arbitrary builtins or imports
    assert calc.evaluate("__import__('os').system('ls')").success is False
    assert calc.evaluate("open('/etc/passwd')").success is False
    assert calc.evaluate("eval('1+1')").success is False
    assert calc.evaluate("x = 5").success is False


def test_division_by_zero():
    calc = SafeFinancialCalculator()
    res = calc.evaluate("10 / 0")
    assert res.success is False
    assert "zero" in res.error.lower()


def test_process_text_calculations():
    calc = SafeFinancialCalculator()
    text = "Revenue grew by [CALC: (150 - 100) / 100 * 100]% YoY."
    processed, audits = calc.process_text_calculations(text)
    assert processed == "Revenue grew by 50% YoY."
    assert len(audits) == 1
    assert audits[0].result == 50.0

    text2 = "Operating margin was:\n```calculation\n(45000 / 100000) * 100\n```%"
    processed2, audits2 = calc.process_text_calculations(text2)
    assert "45%" in processed2
    assert len(audits2) == 1
