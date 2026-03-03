"""
Fía Signals — AI Agent Tools for Python
Crypto market intelligence for LangChain, CrewAI, AutoGen, and any Python AI framework.

Usage:
    from fia_signals_tools import get_market_regime, get_crypto_signals, get_wallet_risk
"""
import requests

BASE_URL = "https://fiasignals.com"
ACP_AGENT_ID = 17266
X402_GATEWAY = "https://x402.fiasignals.com"

def get_market_regime() -> dict:
    """Get current crypto market regime (trending/ranging/volatile/breakout)."""
    r = requests.get(f"{BASE_URL}/v1/intelligence/regime/free", timeout=10)
    return r.json()

def get_crypto_signals(symbol: str = "BTC") -> dict:
    """Get BUY/SELL/HOLD signals with RSI, MACD, ADX for any crypto."""
    r = requests.get(f"{BASE_URL}/v1/dd/quick/{symbol}", timeout=10)
    return r.json()

def get_defi_yields() -> dict:
    """Get best DeFi yields from Aave, Compound, Curve, Lido."""
    r = requests.get(f"{BASE_URL}/v1/yield/rates", timeout=10)
    return r.json()

def get_gas_prices() -> dict:
    """Get real-time gas prices across Ethereum, Polygon, BSC, Arbitrum, Base."""
    r = requests.get(f"{BASE_URL}/v1/gas/prices", timeout=10)
    return r.json()

def get_solana_trending() -> dict:
    """Get trending Solana tokens with volume and risk scores."""
    r = requests.get(f"{BASE_URL}/v1/solana/trending", timeout=10)
    return r.json()

def get_wallet_risk(address: str) -> dict:
    """Get risk score and entity classification for any wallet. (x402 premium)"""
    return {
        "info": "Premium endpoint — requires x402 micropayment",
        "gateway": X402_GATEWAY,
        "endpoint": f"/v1/wallet/risk/{address}",
        "price": "$0.02 USDC",
        "discovery": f"{X402_GATEWAY}/.well-known/x402.json"
    }

# LangChain tool wrappers
try:
    from langchain.tools import tool
    
    @tool
    def fia_market_regime() -> str:
        """Get current cryptocurrency market regime. Returns trending/ranging/volatile/breakout classification with supporting indicators."""
        return str(get_market_regime())
    
    @tool
    def fia_crypto_signals(symbol: str) -> str:
        """Get trading signals for a cryptocurrency. Returns BUY/SELL/HOLD recommendation with RSI, MACD, ADX values. Input: token symbol like BTC, ETH, SOL."""
        return str(get_crypto_signals(symbol))
    
    @tool  
    def fia_defi_yields() -> str:
        """Get best DeFi yield rates across major protocols (Aave, Compound, Curve, Lido)."""
        return str(get_defi_yields())
        
    LANGCHAIN_TOOLS = [fia_market_regime, fia_crypto_signals, fia_defi_yields]
except ImportError:
    LANGCHAIN_TOOLS = []

__version__ = "0.1.0"
__all__ = ["get_market_regime", "get_crypto_signals", "get_defi_yields", "get_gas_prices", "get_solana_trending", "get_wallet_risk", "LANGCHAIN_TOOLS"]
