# Fía Signals MCP Server

[![MCP](https://img.shields.io/badge/MCP-Compatible-blue)](https://modelcontextprotocol.io)
[![x402](https://img.shields.io/badge/x402-Pay--per--call-green)](https://x402.fiasignals.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Professional-grade crypto market intelligence for AI agents. 8 MCP tools backed by real-time data from Binance, DeFiLlama, Etherscan, and more.

## Tools

| Tool | Description | Free? |
|------|-------------|-------|
| `get_market_regime` | Detect trending/ranging/volatile market regime | ✅ |
| `get_crypto_signals` | BUY/SELL/HOLD with RSI, MACD, ADX, volume | ✅ |
| `get_price_levels` | Support/resistance levels + directional bias | ✅ |
| `get_defi_yields` | Best yields across Aave, Compound, Curve, Lido | ✅ |
| `get_solana_tokens` | Trending Solana tokens with risk scores | ✅ |
| `get_wallet_risk` | Wallet risk score + entity classification | x402 |
| `scan_mev` | MEV bot detection + sandwich attack risk | x402 |
| `audit_contract` | Smart contract quick audit (reentrancy, access control) | x402 |

## Quick Start

### With Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fia-signals": {
      "url": "https://fiasignals.com/.well-known/mcp.json"
    }
  }
}
```

### With any MCP client

```bash
npx @fiasignals/mcp-server
```

## x402 Micropayments

Premium tools use x402 pay-per-call (USDC on Solana, $0.001–$0.05/call). Any agent with a Solana wallet can call instantly.

Discovery doc: [x402.fiasignals.com/.well-known/x402.json](https://x402.fiasignals.com/.well-known/x402.json)

Wallet: `GScv2iEvgUHcYyKpbVBHMZU3ELLvhAq4hS9aD75CiErW`

## Virtuals Protocol ACP

Hire as an agent-to-agent service on Virtuals Protocol:

**Agent ID:** 17266 (CryptoIntel)  
[View on ACP Marketplace](https://app.virtuals.io/acp/agent-details/17266)

Offerings:
- `free_crypto_sample` — $0.01
- `crypto_signals` — $0.15  
- `price_prediction` — $0.10
- `blockchain_analysis` — $0.25
- `yield_scanner` — $0.25

## Links

- 🌐 [Website](https://fiasignals.com)
- 📡 [x402 Gateway](https://x402.fiasignals.com)
- 🤖 [ACP Marketplace](https://app.virtuals.io/acp/agent-details/17266)
- 📖 [API Docs](https://fiasignals.com/llms.txt)
- 📊 [OpenAPI](https://fiasignals.com/openapi.json)
