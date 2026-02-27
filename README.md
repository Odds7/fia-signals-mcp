# 🔮 Fia Signals MCP Server

**Real-time crypto market intelligence for AI agents.** Connect your Claude, Cursor, or any MCP-compatible AI to live market data in 60 seconds.

[![MCP Registry](https://img.shields.io/badge/MCP_Registry-Published-brightgreen)](https://registry.modelcontextprotocol.io)
[![x402](https://img.shields.io/badge/x402-32_endpoints-blue)](https://api.fiasignals.com/.well-known/x402.json)
[![API](https://img.shields.io/badge/API-56_routes-orange)](https://api.fiasignals.com/docs)

## Quick Start

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fia-signals": {
      "command": "npx",
      "args": ["-y", "fia-signals-mcp"],
      "env": {}
    }
  }
}
```

### Direct API (no auth needed)

```bash
# Gas prices across 14 chains
curl https://api.fiasignals.com/v1/gas/prices/ethereum

# Solana trending tokens
curl https://api.fiasignals.com/v1/solana/trending

# Quick smart contract audit
curl https://api.fiasignals.com/v1/audit/quick/0xdAC17F958D2ee523a2206206994597C13D831ec7

# Full Swagger docs
open https://api.fiasignals.com/docs
```

## 8 MCP Tools

| Tool | Description |
|------|-------------|
| `get_market_regime` | BTC volatility regime classification (trending/ranging/volatile/breakout) with RSI, ADX, confidence level |
| `get_fear_greed` | Crypto Fear & Greed Index with classification and historical trend |
| `get_funding_rates` | Top perpetual funding rates across Bybit with annualised yield |
| `get_technical_signals` | Multi-indicator technical analysis: EMA, RSI, MACD, Bollinger, ADX |
| `get_prices` | Real-time crypto prices from Binance with 24h change and volume |
| `get_liquidations` | Recent liquidation events and cluster analysis |
| `get_market_dominance` | BTC/ETH dominance percentages and market cap data |
| `get_full_market_brief` | Complete market intelligence brief — all tools in one call |

## REST API — 56 Endpoints

16 AI agent services covering:

- 🔥 **Gas Oracle** — 14 chains including Solana
- 🔍 **Smart Contract Auditor** — automated security analysis
- 🦈 **MEV Scanner** — sandwich attack & bot detection
- 👛 **Wallet Intelligence** — risk scoring & behavior analysis
- ☀️ **Solana Analytics** — token scanner, DeFi toolkit, rug pull detection
- 📚 **Research Synthesis** — academic paper discovery
- 💰 **DeFi Yield Optimizer** — cross-protocol yield comparison
- 🪙 **Token Due Diligence** — fundamental analysis

**Free endpoints** (no auth): gas prices, trending tokens, MEV bots, quick audits, quick DD, yield rates, research topics

**Paid endpoints** (x402 USDC micropayments): $0.001 - $1.00 per call

## x402 Pay-Per-Call

AI agents can pay per API call using USDC on Base:

```
GET /regime → HTTP 402 Payment Required
Agent pays $0.001 USDC via x402 facilitator
GET /regime (with payment proof) → 200 OK + data
```

Discovery: [`api.fiasignals.com/.well-known/x402.json`](https://api.fiasignals.com/.well-known/x402.json)

## Links

- 📋 [API Documentation](https://api.fiasignals.com/docs)
- 🌐 [Website](https://fiasignals.com)
- 📦 [MCP Registry](https://registry.modelcontextprotocol.io)
- 🛒 [ACP Manifest](https://fiasignals.com/.well-known/acp)

## License

MIT
