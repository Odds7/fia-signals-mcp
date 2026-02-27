#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
from mcp.server.fastmcp import FastMCP

DEFAULT_BASE_URL = "https://x402.fiasignals.com"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_CACHE_TTL_SECONDS = 120
DEFAULT_PRICE_SYMBOLS = ["BTC", "ETH", "SOL"]
DEFAULT_BRIEF_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP"]


@dataclass
class CacheEntry:
    data: dict[str, Any]
    expires_at: float


class FiaSignalsClient:
    """Thin API client with timeout + stale-cache fallback behavior."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, CacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._http = httpx.Client(
            timeout=self.timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": "fia-signals-mcp/1.0",
            },
        )

    def close(self) -> None:
        self._http.close()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _build_cache_key(self, path: str, params: dict[str, Any] | None = None) -> str:
        if not params:
            return path
        parts = [f"{k}={params[k]}" for k in sorted(params)]
        return f"{path}?{'&'.join(parts)}"

    def _cache_get(self, key: str, allow_stale: bool = False) -> dict[str, Any] | None:
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if allow_stale or entry.expires_at > time.time():
                return copy.deepcopy(entry.data)
            return None

    def _cache_put(self, key: str, value: dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[key] = CacheEntry(
                data=copy.deepcopy(value),
                expires_at=time.time() + self.cache_ttl_seconds,
            )

    @staticmethod
    def _decode_payment_required(raw_header: str | None) -> dict[str, Any] | None:
        if not raw_header:
            return None
        try:
            padded = raw_header + ("=" * (-len(raw_header) % 4))
            decoded = base64.b64decode(padded).decode("utf-8")
            payload = json.loads(decoded)
            if isinstance(payload, dict):
                return payload
            return None
        except Exception:
            return None

    @staticmethod
    def _extract_payment_example(payment_payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if not payment_payload:
            return None
        example = (
            payment_payload.get("extensions", {})
            .get("bazaar", {})
            .get("info", {})
            .get("output", {})
            .get("example")
        )
        if isinstance(example, dict):
            return example
        return None

    @staticmethod
    def _extract_payment_quote(payment_payload: dict[str, Any] | None) -> dict[str, Any] | None:
        if not payment_payload:
            return None
        accepts = payment_payload.get("accepts")
        if not isinstance(accepts, list) or not accepts:
            return None
        first = accepts[0]
        if not isinstance(first, dict):
            return None
        extra = first.get("extra") if isinstance(first.get("extra"), dict) else {}
        return {
            "scheme": first.get("scheme"),
            "network": first.get("network"),
            "asset": first.get("asset"),
            "amount": first.get("amount"),
            "pay_to": first.get("payTo"),
            "max_timeout_seconds": first.get("maxTimeoutSeconds"),
            "token_name": extra.get("name"),
            "token_version": extra.get("version"),
        }

    def _with_meta(
        self,
        payload: dict[str, Any],
        *,
        source: str,
        path: str,
        payment_required: bool,
        payment_quote: dict[str, Any] | None,
        warning: str | None = None,
    ) -> dict[str, Any]:
        out = dict(payload)
        meta: dict[str, Any] = dict(out.get("_meta") or {})
        meta.update(
            {
                "source": source,
                "path": path,
                "timestamp": self._now_iso(),
                "payment_required": payment_required,
            }
        )
        if payment_quote is not None:
            meta["payment_quote"] = payment_quote
        if warning:
            meta["warning"] = warning
        out["_meta"] = meta
        return out

    def _request_json(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        cache_key: str | None = None,
        fallback_builder: Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        key = cache_key or self._build_cache_key(path, params)
        url = f"{self.base_url}{path}"

        try:
            response = self._http.get(url, params=params)
        except httpx.TimeoutException:
            stale = self._cache_get(key, allow_stale=True)
            if stale is not None:
                return self._with_meta(
                    stale,
                    source="cache_stale_timeout",
                    path=path,
                    payment_required=bool(stale.get("_meta", {}).get("payment_required")),
                    payment_quote=stale.get("_meta", {}).get("payment_quote"),
                    warning="Live API timeout; returned stale cached data.",
                )
            return self._with_meta(
                {"error": "timeout", "message": f"Timed out calling {url}"},
                source="error_timeout",
                path=path,
                payment_required=False,
                payment_quote=None,
            )
        except httpx.HTTPError as exc:
            stale = self._cache_get(key, allow_stale=True)
            if stale is not None:
                return self._with_meta(
                    stale,
                    source="cache_stale_error",
                    path=path,
                    payment_required=bool(stale.get("_meta", {}).get("payment_required")),
                    payment_quote=stale.get("_meta", {}).get("payment_quote"),
                    warning=f"Live API error ({exc.__class__.__name__}); returned stale cached data.",
                )
            return self._with_meta(
                {"error": "request_error", "message": str(exc)},
                source="error_request",
                path=path,
                payment_required=False,
                payment_quote=None,
            )

        payment_payload = self._decode_payment_required(response.headers.get("payment-required"))
        payment_quote = self._extract_payment_quote(payment_payload)

        if response.status_code == 200:
            try:
                body = response.json()
            except json.JSONDecodeError:
                body = {}

            if isinstance(body, dict):
                out = self._with_meta(
                    body,
                    source="live",
                    path=path,
                    payment_required=False,
                    payment_quote=None,
                )
            else:
                out = self._with_meta(
                    {"result": body},
                    source="live",
                    path=path,
                    payment_required=False,
                    payment_quote=None,
                )
            self._cache_put(key, out)
            return out

        if response.status_code == 402:
            example = self._extract_payment_example(payment_payload)
            if example is not None:
                out = self._with_meta(
                    example,
                    source="payment_required_example",
                    path=path,
                    payment_required=True,
                    payment_quote=payment_quote,
                )
                self._cache_put(key, out)
                return out

            if fallback_builder is not None:
                fallback_payload = fallback_builder()
                out = self._with_meta(
                    fallback_payload,
                    source="fallback_preview",
                    path=path,
                    payment_required=True,
                    payment_quote=payment_quote,
                )
                self._cache_put(key, out)
                return out

            stale = self._cache_get(key, allow_stale=True)
            if stale is not None:
                return self._with_meta(
                    stale,
                    source="cache_stale_payment_required",
                    path=path,
                    payment_required=True,
                    payment_quote=payment_quote,
                    warning="Paid endpoint blocked; returned stale cached data.",
                )

            return self._with_meta(
                {"error": "payment_required", "message": "x402 payment required for this endpoint."},
                source="error_payment_required",
                path=path,
                payment_required=True,
                payment_quote=payment_quote,
            )

        stale = self._cache_get(key, allow_stale=True)
        if stale is not None:
            return self._with_meta(
                stale,
                source="cache_stale_http_error",
                path=path,
                payment_required=bool(stale.get("_meta", {}).get("payment_required")),
                payment_quote=stale.get("_meta", {}).get("payment_quote"),
                warning=f"Live API returned HTTP {response.status_code}; returned stale cached data.",
            )

        return self._with_meta(
            {
                "error": "http_error",
                "status_code": response.status_code,
                "body": response.text[:500],
            },
            source="error_http",
            path=path,
            payment_required=False,
            payment_quote=None,
        )

    @staticmethod
    def _as_float(value: Any, default: float | None = None) -> float | None:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int | None = None) -> int | None:
        if value is None:
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_regime(raw_regime: Any) -> str:
        value = str(raw_regime or "").upper().replace(" ", "_")
        if "TREND" in value and "UP" in value:
            return "TRENDING_UP"
        if "TREND" in value and "DOWN" in value:
            return "TRENDING_DOWN"
        if "RANG" in value:
            return "RANGING"
        if value in {"TRENDING_UP", "TRENDING_DOWN", "RANGING"}:
            return value
        return "RANGING"

    @staticmethod
    def _confidence_from_adx(adx: float | None) -> tuple[float, str]:
        if adx is None:
            return 0.5, "MEDIUM"
        score = max(0.3, min(0.95, adx / 45.0))
        if score >= 0.75:
            label = "HIGH"
        elif score >= 0.55:
            label = "MEDIUM"
        else:
            label = "LOW"
        return round(score, 2), label

    @staticmethod
    def _extract_confidence(raw_confidence: Any, adx: float | None) -> tuple[float, str]:
        if isinstance(raw_confidence, (int, float)):
            score = max(0.0, min(1.0, float(raw_confidence)))
            label = "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.55 else "LOW"
            return round(score, 2), label

        if isinstance(raw_confidence, str):
            match = re.search(r"([0-9]*\.?[0-9]+)", raw_confidence)
            if match:
                parsed = float(match.group(1))
                if parsed > 1.0:
                    parsed = parsed / 100.0 if parsed > 10 else parsed
                score = max(0.0, min(1.0, parsed))
                label = raw_confidence.upper().split()[0].strip("(_")
                if label not in {"LOW", "MEDIUM", "HIGH"}:
                    label = "HIGH" if score >= 0.75 else "MEDIUM" if score >= 0.55 else "LOW"
                return round(score, 2), label

        return FiaSignalsClient._confidence_from_adx(adx)

    @staticmethod
    def _regime_recommendation(regime: str, rsi: float | None, adx: float | None) -> str:
        rsi_val = rsi if rsi is not None else 50.0
        if regime == "TRENDING_UP":
            if rsi_val >= 70:
                return "Uptrend intact but stretched; favour pullback entries with tight risk controls."
            return "Trend-following long bias; buy dips while ADX confirms momentum."
        if regime == "TRENDING_DOWN":
            if rsi_val <= 30:
                return "Downtrend is extended; avoid fresh shorts into exhaustion without confirmation."
            return "Defensive posture; favour short setups and reduce long exposure."
        if adx is not None and adx < 20:
            return "Range conditions; prefer mean-reversion trades and smaller position sizes."
        return "Mixed conditions; wait for directional confirmation before sizing up."

    @staticmethod
    def _fear_greed_classification(value: int) -> str:
        if value <= 20:
            return "Extreme Fear"
        if value <= 40:
            return "Fear"
        if value < 60:
            return "Neutral"
        if value < 80:
            return "Greed"
        return "Extreme Greed"

    @staticmethod
    def _fear_greed_trend(value: int) -> str:
        if value <= 20:
            return "FEAR_DEEPENING"
        if value <= 40:
            return "RISK_OFF"
        if value < 60:
            return "SIDEWAYS"
        if value < 80:
            return "RISK_ON"
        return "EUPHORIC"

    @staticmethod
    def _contrarian_signal(value: int) -> str:
        if value <= 20:
            return "CONTRARIAN_BULLISH_WATCH"
        if value >= 80:
            return "CONTRARIAN_BEARISH_WATCH"
        return "NEUTRAL"

    @staticmethod
    def _composite_signal(regime: str, rsi: float | None, fear_greed_value: int | None = None) -> str:
        score = 0
        rsi_val = rsi if rsi is not None else 50.0

        if regime == "TRENDING_UP":
            score += 1
        elif regime == "TRENDING_DOWN":
            score -= 1

        if rsi_val >= 55:
            score += 1
        elif rsi_val <= 45:
            score -= 1

        if fear_greed_value is not None:
            if fear_greed_value <= 20:
                score -= 1
            elif fear_greed_value >= 80:
                score += 1

        if score >= 2:
            return "BULLISH"
        if score <= -2:
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _altcoin_season_signal(
        btc_dominance: float | None,
        eth_dominance: float | None,
        fear_greed_value: int | None = None,
        regime: str | None = None,
    ) -> str:
        if btc_dominance is not None and eth_dominance is not None:
            if btc_dominance <= 50 and eth_dominance >= 18:
                return "ALTCOIN_MOMENTUM"
            if btc_dominance >= 56:
                return "BTC_LED_MARKET"
            return "BALANCED_ROTATION"

        if fear_greed_value is not None and regime == "TRENDING_UP" and fear_greed_value >= 55:
            return "ALT_RISK_ON_WATCH"
        if fear_greed_value is not None and fear_greed_value <= 35:
            return "BTC_DEFENSIVE"
        return "UNDETERMINED"

    def get_preview(self) -> dict[str, Any]:
        return self._request_json("/preview", cache_key="preview")

    def get_market_regime(self) -> dict[str, Any]:
        def fallback_from_preview() -> dict[str, Any]:
            preview = self.get_preview()
            raw = (preview.get("preview") or {}).get("regime") or {}
            rsi = self._as_float(raw.get("rsi"), 50.0)
            adx = self._as_float(raw.get("adx"), 20.0)
            regime = self._normalize_regime(raw.get("regime"))
            confidence, confidence_label = self._confidence_from_adx(adx)
            return {
                "regime": regime,
                "confidence": confidence,
                "confidence_label": confidence_label,
                "rsi": rsi,
                "adx": adx,
                "recommendation": self._regime_recommendation(regime, rsi, adx),
            }

        raw = self._request_json("/regime", cache_key="regime", fallback_builder=fallback_from_preview)
        meta = dict(raw.get("_meta") or {})

        regime = self._normalize_regime(raw.get("regime"))
        rsi = self._as_float(raw.get("rsi"), 50.0)
        adx = self._as_float(raw.get("adx"), 20.0)
        confidence, confidence_label = self._extract_confidence(raw.get("confidence"), adx)
        recommendation = raw.get("recommendation") or self._regime_recommendation(regime, rsi, adx)

        return {
            "regime": regime,
            "confidence": confidence,
            "confidence_label": confidence_label,
            "rsi": rsi,
            "adx": adx,
            "recommendation": recommendation,
            "_meta": meta,
        }

    def get_fear_greed(self) -> dict[str, Any]:
        def fallback_from_preview() -> dict[str, Any]:
            preview = self.get_preview()
            raw = (preview.get("preview") or {}).get("fear_greed") or {}
            value = self._as_int(raw.get("value"), 50) or 50
            classification = raw.get("classification") or self._fear_greed_classification(value)
            return {
                "value": value,
                "classification": classification,
                "trend_7d": self._fear_greed_trend(value),
                "contrarian_signal": self._contrarian_signal(value),
            }

        raw = self._request_json("/fear-greed", cache_key="fear_greed", fallback_builder=fallback_from_preview)
        meta = dict(raw.get("_meta") or {})

        value = self._as_int(raw.get("value") or raw.get("index"), 50) or 50
        classification = raw.get("classification") or self._fear_greed_classification(value)
        trend = (
            raw.get("trend_7d")
            or raw.get("trend")
            or raw.get("seven_day_trend")
            or self._fear_greed_trend(value)
        )
        contrarian = raw.get("contrarian_signal") or self._contrarian_signal(value)

        return {
            "value": value,
            "classification": classification,
            "trend_7d": trend,
            "contrarian_signal": contrarian,
            "_meta": meta,
        }

    def get_funding_rates(self, top_n: int = 10) -> dict[str, Any]:
        safe_top_n = max(1, min(int(top_n), 50))
        raw = self._request_json(
            "/funding",
            params={"top_n": safe_top_n},
            cache_key=f"funding:{safe_top_n}",
            fallback_builder=lambda: {"count": 0, "results": []},
        )
        meta = dict(raw.get("_meta") or {})

        rows = raw.get("results")
        if not isinstance(rows, list):
            rows = []

        formatted: list[dict[str, Any]] = []
        for row in rows[:safe_top_n]:
            if not isinstance(row, dict):
                continue
            rate_pct = self._as_float(row.get("funding_rate_pct") or row.get("fundingRatePct"), 0.0) or 0.0
            annualized_pct = round(rate_pct * 3 * 365, 4)
            direction = row.get("direction")
            if not direction:
                direction = "positive" if rate_pct > 0 else "negative" if rate_pct < 0 else "flat"

            formatted.append(
                {
                    "symbol": row.get("symbol"),
                    "funding_rate_pct": rate_pct,
                    "annualized_pct": annualized_pct,
                    "direction": direction,
                    "next_funding_time": row.get("next_funding_time") or row.get("nextFundingTime"),
                }
            )

        return {
            "count": len(formatted),
            "results": formatted,
            "timestamp": raw.get("timestamp") or self._now_iso(),
            "_meta": meta,
        }

    def get_technical_signals(self, symbol: str = "BTCUSDT") -> dict[str, Any]:
        symbol_norm = (symbol or "BTCUSDT").upper()

        def fallback_from_preview() -> dict[str, Any]:
            regime = self.get_market_regime()
            fear_greed = self.get_fear_greed()
            fallback_rsi = regime.get("rsi") if symbol_norm.startswith("BTC") else 50.0
            fallback_rsi_val = self._as_float(fallback_rsi, 50.0) or 50.0
            return {
                "symbol": symbol_norm,
                "rsi_14": round(fallback_rsi_val, 2),
                "macd": {"line": 0.0, "signal": 0.0, "histogram": 0.0},
                "bollinger_percent_b": 0.5,
                "composite_signal": self._composite_signal(
                    regime=regime.get("regime", "RANGING"),
                    rsi=fallback_rsi_val,
                    fear_greed_value=fear_greed.get("value"),
                ),
                "interval": "1h",
                "note": "Derived from preview data because /signals requires x402 payment.",
            }

        raw = self._request_json(
            "/signals",
            params={"symbol": symbol_norm},
            cache_key=f"signals:{symbol_norm}",
            fallback_builder=fallback_from_preview,
        )
        meta = dict(raw.get("_meta") or {})

        rsi_14 = self._as_float(raw.get("rsi_14") or raw.get("rsi"), 50.0) or 50.0

        macd_payload = raw.get("macd") if isinstance(raw.get("macd"), dict) else {}
        macd_line = self._as_float(macd_payload.get("line") or raw.get("macd_line"), 0.0) or 0.0
        macd_signal = self._as_float(macd_payload.get("signal") or raw.get("macd_signal"), 0.0) or 0.0
        macd_hist = self._as_float(macd_payload.get("histogram") or raw.get("macd_histogram"), 0.0) or 0.0

        percent_b = self._as_float(
            raw.get("bollinger_percent_b") or raw.get("bb_percent_b") or raw.get("percent_b"),
            0.5,
        )
        percent_b_val = percent_b if percent_b is not None else 0.5

        fear_greed = self.get_fear_greed()
        regime = self.get_market_regime()
        composite = raw.get("composite_signal") or self._composite_signal(
            regime=regime.get("regime", "RANGING"),
            rsi=rsi_14,
            fear_greed_value=fear_greed.get("value"),
        )

        return {
            "symbol": symbol_norm,
            "rsi_14": round(rsi_14, 2),
            "macd": {
                "line": round(macd_line, 6),
                "signal": round(macd_signal, 6),
                "histogram": round(macd_hist, 6),
            },
            "bollinger_percent_b": round(percent_b_val, 4),
            "composite_signal": composite,
            "interval": raw.get("interval") or "1h",
            "_meta": meta,
        }

    def get_prices(self, symbols: list[str] | None = None) -> dict[str, Any]:
        symbol_list = symbols or list(DEFAULT_PRICE_SYMBOLS)
        cleaned = [s.upper().strip() for s in symbol_list if isinstance(s, str) and s.strip()]
        cleaned = cleaned[:20] if cleaned else list(DEFAULT_PRICE_SYMBOLS)
        joined = ",".join(cleaned)

        raw = self._request_json(
            "/prices",
            params={"symbols": joined},
            cache_key=f"prices:{joined}",
            fallback_builder=lambda: {"count": 0, "results": []},
        )
        meta = dict(raw.get("_meta") or {})

        rows = raw.get("results")
        if not isinstance(rows, list):
            rows = []

        formatted: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            formatted.append(
                {
                    "symbol": row.get("symbol"),
                    "price": self._as_float(row.get("price")),
                    "change_24h_pct": self._as_float(row.get("change_24h_pct") or row.get("change24h")),
                    "high_24h": self._as_float(row.get("high_24h") or row.get("high24h")),
                    "low_24h": self._as_float(row.get("low_24h") or row.get("low24h")),
                }
            )

        return {
            "count": len(formatted),
            "results": formatted,
            "timestamp": raw.get("timestamp") or self._now_iso(),
            "_meta": meta,
        }

    def get_liquidations(self) -> dict[str, Any]:
        raw = self._request_json(
            "/liquidations",
            cache_key="liquidations",
            fallback_builder=lambda: {
                "events": [],
                "long_volume_usd": None,
                "short_volume_usd": None,
                "largest_single_liquidation": None,
                "summary": "Liquidation feed requires x402 payment.",
            },
        )
        meta = dict(raw.get("_meta") or {})

        events = raw.get("events") if isinstance(raw.get("events"), list) else []
        if not events and isinstance(raw.get("results"), list):
            events = raw.get("results")

        long_volume = self._as_float(
            raw.get("long_volume_usd")
            or raw.get("long_liquidations_usd")
            or raw.get("longs_usd")
        )
        short_volume = self._as_float(
            raw.get("short_volume_usd")
            or raw.get("short_liquidations_usd")
            or raw.get("shorts_usd")
        )

        largest = raw.get("largest_single_liquidation")
        if largest is None and isinstance(raw.get("largest_liquidation"), dict):
            largest = raw.get("largest_liquidation")

        if largest is None and events:
            numeric_events = [
                e for e in events if isinstance(e, dict) and self._as_float(e.get("size_usd") or e.get("value_usd"))
            ]
            if numeric_events:
                largest = max(
                    numeric_events,
                    key=lambda e: self._as_float(e.get("size_usd") or e.get("value_usd"), 0.0) or 0.0,
                )

        return {
            "events": events,
            "long_volume_usd": long_volume,
            "short_volume_usd": short_volume,
            "largest_single_liquidation": largest,
            "_meta": meta,
        }

    def get_market_dominance(self) -> dict[str, Any]:
        def fallback_from_preview() -> dict[str, Any]:
            fear_greed = self.get_fear_greed()
            regime = self.get_market_regime()
            signal = self._altcoin_season_signal(
                btc_dominance=None,
                eth_dominance=None,
                fear_greed_value=fear_greed.get("value"),
                regime=regime.get("regime"),
            )
            return {
                "btc_dominance_pct": None,
                "eth_dominance_pct": None,
                "altcoin_season_signal": signal,
                "note": "Dominance percentages require x402 payment; signal inferred from regime/sentiment.",
            }

        raw = self._request_json(
            "/dominance",
            cache_key="dominance",
            fallback_builder=fallback_from_preview,
        )
        meta = dict(raw.get("_meta") or {})

        btc_dom = self._as_float(raw.get("btc_dominance_pct") or raw.get("btc_dominance") or raw.get("btc"))
        eth_dom = self._as_float(raw.get("eth_dominance_pct") or raw.get("eth_dominance") or raw.get("eth"))

        fear_greed = self.get_fear_greed()
        regime = self.get_market_regime()
        signal = raw.get("altcoin_season_signal") or raw.get("altseason_signal") or self._altcoin_season_signal(
            btc_dominance=btc_dom,
            eth_dominance=eth_dom,
            fear_greed_value=fear_greed.get("value"),
            regime=regime.get("regime"),
        )

        return {
            "btc_dominance_pct": btc_dom,
            "eth_dominance_pct": eth_dom,
            "altcoin_season_signal": signal,
            "_meta": meta,
        }

    def get_full_market_brief(self) -> dict[str, Any]:
        regime = self.get_market_regime()
        fear_greed = self.get_fear_greed()
        prices = self.get_prices(DEFAULT_BRIEF_SYMBOLS)
        funding = self.get_funding_rates(5)

        funding_rows = funding.get("results") if isinstance(funding.get("results"), list) else []
        avg_funding = None
        if funding_rows:
            vals = [r.get("funding_rate_pct") for r in funding_rows if isinstance(r, dict)]
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            if vals:
                avg_funding = round(sum(vals) / len(vals), 6)

        return {
            "regime": {
                "regime": regime.get("regime"),
                "confidence": regime.get("confidence"),
                "rsi": regime.get("rsi"),
                "adx": regime.get("adx"),
                "recommendation": regime.get("recommendation"),
            },
            "fear_greed": {
                "value": fear_greed.get("value"),
                "classification": fear_greed.get("classification"),
                "trend_7d": fear_greed.get("trend_7d"),
                "contrarian_signal": fear_greed.get("contrarian_signal"),
            },
            "top_prices": (prices.get("results") or [])[:5],
            "funding_summary": {
                "count": funding.get("count"),
                "average_funding_rate_pct": avg_funding,
                "top_rates": funding_rows[:5],
            },
            "timestamp": self._now_iso(),
            "_meta": {
                "sources": {
                    "regime": regime.get("_meta", {}).get("source"),
                    "fear_greed": fear_greed.get("_meta", {}).get("source"),
                    "prices": prices.get("_meta", {}).get("source"),
                    "funding": funding.get("_meta", {}).get("source"),
                }
            },
        }


_default_client = FiaSignalsClient(
    base_url=os.getenv("FIA_BASE_URL", DEFAULT_BASE_URL),
    timeout_seconds=float(os.getenv("FIA_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
    cache_ttl_seconds=int(os.getenv("FIA_CACHE_TTL_SECONDS", DEFAULT_CACHE_TTL_SECONDS)),
)


def get_market_regime(client: FiaSignalsClient | None = None) -> dict[str, Any]:
    return (client or _default_client).get_market_regime()


def get_fear_greed(client: FiaSignalsClient | None = None) -> dict[str, Any]:
    return (client or _default_client).get_fear_greed()


def get_funding_rates(top_n: int = 10, client: FiaSignalsClient | None = None) -> dict[str, Any]:
    return (client or _default_client).get_funding_rates(top_n=top_n)


def get_technical_signals(
    symbol: str = "BTCUSDT", client: FiaSignalsClient | None = None
) -> dict[str, Any]:
    return (client or _default_client).get_technical_signals(symbol=symbol)


def get_prices(symbols: list[str] | None = None, client: FiaSignalsClient | None = None) -> dict[str, Any]:
    return (client or _default_client).get_prices(symbols=symbols)


def get_liquidations(client: FiaSignalsClient | None = None) -> dict[str, Any]:
    return (client or _default_client).get_liquidations()


def get_market_dominance(client: FiaSignalsClient | None = None) -> dict[str, Any]:
    return (client or _default_client).get_market_dominance()


def get_full_market_brief(client: FiaSignalsClient | None = None) -> dict[str, Any]:
    return (client or _default_client).get_full_market_brief()


def market_briefing_prompt(client: FiaSignalsClient | None = None) -> str:
    active = client or _default_client
    regime = active.get_market_regime()
    fear_greed = active.get_fear_greed()
    btc_price = active.get_prices(["BTC"])  # type: ignore[arg-type]
    btc = (btc_price.get("results") or [{}])[0]

    return (
        "You are a crypto trading analyst. "
        "Here is the current market data: "
        f"regime={regime.get('regime')} (confidence={regime.get('confidence')}, RSI={regime.get('rsi')}, ADX={regime.get('adx')}), "
        f"fear_greed={fear_greed.get('value')} ({fear_greed.get('classification')}), "
        f"btc_price={btc.get('price')}. "
        "Provide a concise trading outlook with key risks and invalidation levels."
    )


def trading_decision_prompt(symbol: str, side: str, client: FiaSignalsClient | None = None) -> str:
    active = client or _default_client
    symbol_norm = (symbol or "BTCUSDT").upper()
    side_norm = (side or "BUY").upper()

    regime = active.get_market_regime()
    fear_greed = active.get_fear_greed()
    technical = active.get_technical_signals(symbol_norm)

    return (
        f"Analyse whether to {side_norm} {symbol_norm} given: "
        f"regime={regime.get('regime')}, RSI={technical.get('rsi_14')}, "
        f"F&G={fear_greed.get('value')} ({fear_greed.get('classification')}). "
        "Use the Pete Protocol: only enter if RSI 55-70, regime trending, "
        "and Fear & Greed is not in extreme fear for 3+ days. "
        "Return decision, confidence, invalidation level, and one-line risk summary."
    )


def create_mcp_server(
    client: FiaSignalsClient,
    *,
    host: str,
    port: int,
    log_level: str = "INFO",
) -> FastMCP:
    mcp = FastMCP(
        name="fia-signals",
        instructions=(
            "Fía Signals crypto market intelligence for AI agents. "
            "Use these tools for regime, sentiment, technicals, prices, funding, liquidations, and market briefs."
        ),
        website_url="https://fiasignals.com",
        host=host,
        port=port,
        log_level=log_level.upper(),
    )

    @mcp.tool(name="get_market_regime")
    def tool_get_market_regime() -> dict[str, Any]:
        return client.get_market_regime()

    @mcp.tool(name="get_fear_greed")
    def tool_get_fear_greed() -> dict[str, Any]:
        return client.get_fear_greed()

    @mcp.tool(name="get_funding_rates")
    def tool_get_funding_rates(top_n: int = 10) -> dict[str, Any]:
        return client.get_funding_rates(top_n=top_n)

    @mcp.tool(name="get_technical_signals")
    def tool_get_technical_signals(symbol: str = "BTCUSDT") -> dict[str, Any]:
        return client.get_technical_signals(symbol=symbol)

    @mcp.tool(name="get_prices")
    def tool_get_prices(symbols: list[str] | None = None) -> dict[str, Any]:
        return client.get_prices(symbols=symbols)

    @mcp.tool(name="get_liquidations")
    def tool_get_liquidations() -> dict[str, Any]:
        return client.get_liquidations()

    @mcp.tool(name="get_market_dominance")
    def tool_get_market_dominance() -> dict[str, Any]:
        return client.get_market_dominance()

    @mcp.tool(name="get_full_market_brief")
    def tool_get_full_market_brief() -> dict[str, Any]:
        return client.get_full_market_brief()

    @mcp.resource("fiasignals://market/regime", name="market-regime")
    def resource_market_regime() -> dict[str, Any]:
        return client.get_market_regime()

    @mcp.resource("fiasignals://market/sentiment", name="market-sentiment")
    def resource_market_sentiment() -> dict[str, Any]:
        return client.get_fear_greed()

    @mcp.resource("fiasignals://market/prices/{symbol}", name="market-price")
    def resource_market_price(symbol: str) -> dict[str, Any]:
        result = client.get_prices([symbol])
        rows = result.get("results") if isinstance(result.get("results"), list) else []
        first = rows[0] if rows else {}
        return {
            "symbol": (symbol or "").upper(),
            "price": first.get("price"),
            "change_24h_pct": first.get("change_24h_pct"),
            "high_24h": first.get("high_24h"),
            "low_24h": first.get("low_24h"),
            "_meta": result.get("_meta"),
        }

    @mcp.resource("fiasignals://signals/{symbol}", name="technical-signals")
    def resource_signal(symbol: str) -> dict[str, Any]:
        return client.get_technical_signals(symbol)

    @mcp.prompt(name="market_briefing_prompt")
    def prompt_market_briefing() -> str:
        return market_briefing_prompt(client)

    @mcp.prompt(name="trading_decision_prompt")
    def prompt_trading_decision(symbol: str, side: str) -> str:
        return trading_decision_prompt(symbol=symbol, side=side, client=client)

    return mcp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fía Signals MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode: stdio or http (SSE).",
    )
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8403")))
    parser.add_argument("--log-level", default=os.getenv("MCP_LOG_LEVEL", "INFO"))
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("FIA_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))),
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--cache-ttl",
        type=int,
        default=int(os.getenv("FIA_CACHE_TTL_SECONDS", str(DEFAULT_CACHE_TTL_SECONDS))),
        help="Cache TTL in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = FiaSignalsClient(
        base_url=os.getenv("FIA_BASE_URL", DEFAULT_BASE_URL),
        timeout_seconds=args.timeout,
        cache_ttl_seconds=args.cache_ttl,
    )
    mcp = create_mcp_server(
        client=client,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )

    transport = "stdio" if args.transport == "stdio" else "sse"
    try:
        mcp.run(transport=transport)
    finally:
        client.close()


if __name__ == "__main__":
    main()
