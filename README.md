# Fía Signals MCP Server

> AI-native crypto market intelligence for AI agents. Real-time regime detection, funding rates, liquidation zones, and more.

[![MCP](https://img.shields.io/badge/MCP-compatible-blue)](https://modelcontextprotocol.io)
[![x402](https://img.shields.io/badge/x402-USDC%20payments-green)](https://x402.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What is this?

An MCP server that gives any AI agent (Claude, ChatGPT, Cursor, etc.) access to professional-grade crypto market intelligence:

- **Market Regime Detection** — HMM-based classification: trending, ranging, volatile, breakout
- **Fear & Greed Index** — Real-time sentiment with classification
- **Funding Rates** — Top perpetual funding rates with annualised yields
- **Technical Signals** — EMA, RSI, MACD, Bollinger, ADX across any symbol
- **Live Prices** — Real-time from Binance with 24h change and volume
- **Liquidation Data** — Recent liquidation events and cluster analysis
- **Market Dominance** — BTC/ETH dominance and total market cap
- **Full Market Brief** — Everything above in one call

## Quick Start

### Claude Desktop (stdio)

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fia-signals": {
      "command": "python3",
      "args": ["/path/to/server.py"]
    }
  }
}
```

### HTTP/SSE (remote agents)

```bash
python3 server.py --transport http --port 8403 --host 0.0.0.0
```

## Available Tools

| Tool | Description | Data Source |
|------|-------------|------------|
| `get_market_regime` | BTC volatility regime with RSI, ADX, confidence | Binance + Bybit |
| `get_fear_greed` | Crypto Fear & Greed Index | alternative.me |
| `get_funding_rates` | Top N perpetual funding rates | Bybit |
| `get_technical_signals` | Multi-indicator technical analysis | Binance |
| `get_prices` | Real-time prices with 24h stats | Binance |
| `get_liquidations` | Recent liquidation events | Bybit |
| `get_market_dominance` | BTC/ETH dominance % | CoinGecko |
| `get_full_market_brief` | Complete market brief (all above) | All sources |

## x402 Micropayments

This server is also available as a pay-per-call API via [x402](https://x402.org) at:

```
https://x402.fiasignals.com
```

34 endpoints available, priced $0.001-$0.005 USDC per call on Base L2. Free preview at `/preview`.

## Also Available On

- **ClaWHub**: `clawhub install fia-signals-skill` — [View on ClaWHub](https://clawhub.com/skills/k97857c0r21etwmwfvw4ab8f8h81yvja)
- **ACP**: `https://fiasignals.com/.well-known/acp` — ChatGPT agent discovery
- **x402 Bazaar**: `https://x402.fiasignals.com/.well-known/x402.json`

## Requirements

- Python 3.11+
- `mcp` package
- `httpx` package

```bash
pip install mcp httpx
```

## License

MIT

## Contact

- **Email**: fia-trading@agentmail.to
- **Website**: https://fiasignals.com
- **Telegram**: [@fiaandgreed](https://t.me/fiaandgreed)
