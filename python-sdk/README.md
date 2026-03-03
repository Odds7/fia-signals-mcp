# Fía Signals Tools — Crypto Intelligence for AI Agents

[![PyPI version](https://badge.fury.io/py/fia-signals-tools.svg)](https://badge.fury.io/py/fia-signals-tools)

Crypto market intelligence for LangChain, CrewAI, AutoGen, and any Python AI agent framework.

## Install

```bash
pip install fia-signals-tools
```

## Quick Start

```python
from fia_signals_tools import get_market_regime, get_crypto_signals, get_defi_yields

# Is the market trending, ranging, or volatile?
regime = get_market_regime()
print(regime)

# Get signals for any crypto
signals = get_crypto_signals("ETH")
print(signals)  # {"signal": "BUY", "rsi": 52, "adx": 28, ...}

# Best DeFi yields
yields = get_defi_yields()
```

## LangChain Integration

```python
from fia_signals_tools import LANGCHAIN_TOOLS

# Add to your agent
agent = initialize_agent(
    tools=LANGCHAIN_TOOLS,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
```

## Premium Tools (x402 Micropayments)
Wallet risk scoring, MEV detection, smart contract auditing — available via [x402 pay-per-call](https://x402.fiasignals.com).

## Links
- [Website](https://fiasignals.com) | [x402 Gateway](https://x402.fiasignals.com) | [ACP Agent #17266](https://app.virtuals.io/acp/agent-details/17266)
