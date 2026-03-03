from setuptools import setup, find_packages

setup(
    name="fia-signals-tools",
    version="0.1.0",
    description="Crypto market intelligence tools for AI agents — market regime, trading signals, DeFi yields, wallet risk scoring",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Fía Signals",
    author_email="fia-trading@agentmail.to",
    url="https://fiasignals.com",
    packages=find_packages(),
    install_requires=["requests>=2.28.0"],
    extras_require={
        "langchain": ["langchain>=0.1.0"],
        "crewai": ["crewai>=0.1.0"],
    },
    keywords=["crypto", "trading", "ai-agents", "langchain", "crewai", "mcp", "defi", "blockchain", "market-intelligence", "x402"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
)
