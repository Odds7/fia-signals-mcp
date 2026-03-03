# Fía Signals MCP Server

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![x402](https://img.shields.io/badge/x402-Pay--per--call-green)](https://x402.fiasignals.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Professional-grade crypto market intelligence for AI agents. 8 MCP tools backed by real-time data from Binance, DeFiLlama, Etherscan, and more.

## Tools

| Tool | Description | Free? |
|------|-------------|-------|
| `get_market_regime` | Detect trending/ranging/volatile/breakout market regime | ✅ Free |
| `get_fear_greed` | Crypto Fear & Greed Index with 7-day history | ✅ Free |
| `get_funding_rates` | Top perpetual funding rates across exchanges | ✅ Free |
| `get_defi_yields` | Best yields across Aave, Compound, Curve, Lido | ✅ Free |
| `get_solana_tokens` | Trending Solana tokens with risk scores | ✅ Free |
| `get_crypto_signals` | BUY/SELL/HOLD with RSI, MACD, ADX | ✅ Free |
| `get_wallet_risk` | Wallet risk score + entity classification | x402 $0.02 |
| `audit_contract` | Smart contract quick audit (reentrancy, access control) | x402 $0.05 |

## Quick Start

### Claude Desktop

```json
{
  "mcpServers": {
    "fia-signals": {
      "url": "https://fiasignals.com/.well-known/mcp.json"
    }
  }
}
```

### Cursor / Windsurf

Add to your MCP config:
```
https://fiasignals.com/.well-known/mcp.json
```

### Python (LangChain / CrewAI / AutoGen)

```bash
pip install fia-signals-tools
```

```python
from fia_signals_tools import get_market_regime, get_crypto_signals, LANGCHAIN_TOOLS

# Standalone
regime = get_market_regime()  # {"regime": "TRENDING_UP", "confidence": 0.8, ...}
signals = get_crypto_signals("ETH")  # {"signal": "BUY", "rsi": 52, ...}

# LangChain agent
from langchain.agents import initialize_agent
agent = initialize_agent(tools=LANGCHAIN_TOOLS, llm=llm)
```

## x402 Micropayments

Premium tools (wallet risk, smart contract audit) use x402 pay-per-call. Any agent with a Solana USDC wallet can call instantly — no API keys, no setup.

- **Discovery:** https://x402.fiasignals.com/.well-known/x402.json
- **Wallet:** `GScv2iEvgUHcYyKpbVBHMZU3ELLvhAq4hS9aD75CiErW`
- **Price:** $0.001–$0.50/call

## Virtuals Protocol ACP (Agent-to-Agent Hiring)

Hire CryptoIntel directly via the Virtuals ACP marketplace. Agent-to-agent escrow payments.

**Agent ID:** 17266  
[→ View on ACP Marketplace](https://app.virtuals.io/acp/agent-details/17266)

| Offering | Price | Description |
|----------|-------|-------------|
| `free_crypto_sample` | $0.01 | Regime + BTC price + gas snapshot |
| `crypto_signals` | $0.15 | BUY/SELL/HOLD with RSI, ADX, MACD |
| `price_prediction` | $0.10 | Support/resistance + directional bias |
| `blockchain_analysis` | $0.25 | Token metrics + on-chain data |
| `yield_scanner` | $0.25 | Best DeFi yields across protocols |

## Links

| Resource | URL |
|----------|-----|
| 🌐 Website | https://fiasignals.com |
| 📡 x402 Gateway | https://x402.fiasignals.com |
| 🤖 ACP Marketplace | https://app.virtuals.io/acp/agent-details/17266 |
| 📖 llms.txt | https://fiasignals.com/llms.txt |
| 📊 OpenAPI | https://fiasignals.com/openapi.json |
| 🔌 MCP Discovery | https://fiasignals.com/.well-known/mcp.json |
