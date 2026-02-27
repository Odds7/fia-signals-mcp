#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable

from server import (
    get_fear_greed,
    get_funding_rates,
    get_full_market_brief,
    get_liquidations,
    get_market_dominance,
    get_market_regime,
    get_prices,
    get_technical_signals,
)


def ensure_dict(name: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise AssertionError(f"{name}: expected dict, got {type(payload).__name__}")
    return payload


def ensure_keys(name: str, payload: dict[str, Any], required: list[str]) -> None:
    missing = [k for k in required if k not in payload]
    if missing:
        raise AssertionError(f"{name}: missing keys {missing}")


def print_payload(name: str, payload: dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def run_test(name: str, fn: Callable[[], Any], required_keys: list[str]) -> tuple[bool, str]:
    try:
        payload = ensure_dict(name, fn())
        ensure_keys(name, payload, required_keys)
        if "error" in payload:
            raise AssertionError(f"{name}: returned error={payload['error']}")
        print_payload(name, payload)
        return True, "ok"
    except Exception as exc:
        print(f"\n=== {name} FAILED ===")
        print(str(exc))
        traceback.print_exc()
        return False, str(exc)


def main() -> int:
    tests: list[tuple[str, Callable[[], Any], list[str]]] = [
        (
            "get_market_regime",
            lambda: get_market_regime(),
            ["regime", "confidence", "rsi", "adx", "recommendation"],
        ),
        (
            "get_fear_greed",
            lambda: get_fear_greed(),
            ["value", "classification", "trend_7d", "contrarian_signal"],
        ),
        (
            "get_funding_rates",
            lambda: get_funding_rates(top_n=10),
            ["count", "results", "timestamp"],
        ),
        (
            "get_technical_signals",
            lambda: get_technical_signals(symbol="BTCUSDT"),
            ["symbol", "rsi_14", "macd", "bollinger_percent_b", "composite_signal"],
        ),
        (
            "get_prices",
            lambda: get_prices(symbols=["BTC", "ETH", "SOL"]),
            ["count", "results", "timestamp"],
        ),
        (
            "get_liquidations",
            lambda: get_liquidations(),
            ["events", "long_volume_usd", "short_volume_usd", "largest_single_liquidation"],
        ),
        (
            "get_market_dominance",
            lambda: get_market_dominance(),
            ["btc_dominance_pct", "eth_dominance_pct", "altcoin_season_signal"],
        ),
        (
            "get_full_market_brief",
            lambda: get_full_market_brief(),
            ["regime", "fear_greed", "top_prices", "funding_summary", "timestamp"],
        ),
    ]

    passed = 0
    failed = 0

    for name, fn, required in tests:
        ok, _ = run_test(name, fn, required)
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n============================")
    print(f"Total: {passed + failed} | Passed: {passed} | Failed: {failed}")
    print("============================")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
