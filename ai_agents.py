"""
AI Agents for Value Investment Analysis using the Claude Agent SDK.

This module provides specialized agents with:
- Subagent dispatch via Claude's Task tool
- Web search via Tavily for real-time data
- News API for market updates
- In-process MCP server exposing screening, valuation, and anomaly tools

Public API is preserved for backward compatibility with app.py:
    ValueInvestmentAgent, ScreeningAgent, AnomalyAgent, ResearchAgent
    AgentResponse
    chat(), chat_async(), reset_conversation()
    quick_screen(), quick_analyze(), run_full_analysis()
"""
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Claude Agent SDK imports
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AgentDefinition,
    tool,
    create_sdk_mcp_server,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ResultMessage,
)

# Optional: Tavily for web search
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = bool(os.getenv("TAVILY_API_KEY"))
except ImportError:
    TAVILY_AVAILABLE = False

# Optional: httpx for news / price APIs
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


DEFAULT_MODEL = "claude-sonnet-4-6"


# ============================================================================
# Tool Functions - Decorated with @tool for Claude Agent SDK MCP server
#
# Tool signature: async def name(args: dict) -> dict
# Return shape:   {"content": [{"type": "text", "text": <json_string>}]}
# ============================================================================

def _ok(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Build a successful MCP tool response."""
    return {"content": [{"type": "text", "text": json.dumps(payload, default=str)}]}


def _err(message: str) -> Dict[str, Any]:
    """Build an error MCP tool response."""
    return {"content": [{"type": "text", "text": json.dumps({"success": False, "error": message})}]}


@tool(
    "screen_stocks",
    "Screen stocks based on fundamental value-investing criteria for a market.",
    {
        "type": "object",
        "properties": {
            "market": {"type": "string", "description": "'US' or 'SG'"},
            "gross_margin_min": {"type": "number", "description": "Minimum gross margin %"},
            "net_margin_min": {"type": "number", "description": "Minimum net margin %"},
            "roe_min": {"type": "number", "description": "Minimum return on equity %"},
            "roa_min": {"type": "number", "description": "Minimum return on assets %"},
            "debt_equity_max": {"type": "number", "description": "Maximum debt-to-equity ratio"},
            "fcf_margin_min": {"type": "number", "description": "Minimum free cash flow margin %"},
            "roic_wacc_min": {"type": "number", "description": "Minimum ROIC minus WACC"},
        },
        "required": ["market"],
    },
)
async def screen_stocks(args: Dict[str, Any]) -> Dict[str, Any]:
    from screener import StockScreener
    from data_loader import DataLoader

    try:
        loader = DataLoader()
        screener = StockScreener(loader)

        criteria: Dict[str, float] = {}
        mapping = {
            "gross_margin_min": "gross_margin",
            "net_margin_min": "net_margin",
            "roe_min": "roe",
            "roa_min": "roa",
            "debt_equity_max": "debt_to_equity",
            "fcf_margin_min": "fcf_margin",
            "roic_wacc_min": "roic_wacc",
        }
        for arg_key, criterion_key in mapping.items():
            if args.get(arg_key) is not None:
                criteria[criterion_key] = args[arg_key]

        results = screener.screen(market=args["market"], criteria=criteria)

        return _ok({
            "success": True,
            "total_matches": len(results),
            "criteria_used": criteria,
            "top_stocks": results.head(20).to_dict("records"),
            "valuation_summary": results["Valuation"].value_counts().to_dict()
            if "Valuation" in results.columns else {},
        })
    except Exception as e:
        return _err(str(e))


@tool(
    "analyze_valuation",
    "Analyze stock valuation using EPV vs Market Cap. Returns Undervalued/Fair/Overvalued.",
    {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "epv": {"type": "number", "description": "Earnings Power Value"},
            "market_cap": {"type": "number", "description": "Market capitalization in millions"},
        },
        "required": ["symbol", "epv", "market_cap"],
    },
)
async def analyze_valuation(args: Dict[str, Any]) -> Dict[str, Any]:
    from valuation import Valuator

    valuator = Valuator()
    status, ratio, margin = valuator.classify(args["epv"], args["market_cap"])

    interpretation = ""
    if status and ratio:
        if status.value == "Undervalued":
            interpretation = f"Stock appears undervalued with {margin:.1f}% margin of safety."
        elif status.value == "Fair Value":
            interpretation = f"Stock is fairly valued. EPV is approximately {ratio:.1%} of market cap."
        elif status.value == "Overvalued":
            interpretation = f"Stock appears overvalued. EPV is only {ratio:.1%} of market cap."

    return _ok({
        "symbol": args["symbol"],
        "epv": args["epv"],
        "market_cap": args["market_cap"],
        "epv_mc_ratio": ratio,
        "valuation_status": status.value if status else "N/A",
        "margin_of_safety_pct": margin,
        "interpretation": interpretation,
    })


@tool(
    "detect_anomalies",
    "Detect financial anomalies (M-Score, Z-Score, F-Score, one-off events) for a company.",
    {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., DDI, TEX, CHCI)"},
        },
        "required": ["symbol"],
    },
)
async def detect_anomalies(args: Dict[str, Any]) -> Dict[str, Any]:
    from anomaly_detector import AnomalyDetector
    from data_loader import DataLoader

    try:
        loader = DataLoader()
        detector = AnomalyDetector(loader)
        report = detector.analyze(args["symbol"])

        anomalies_summary = [
            {
                "category": a.category,
                "severity": a.severity.value,
                "description": a.description,
                "year": a.year,
                "details": a.details,
            }
            for a in report.anomalies[:15]
        ]

        return _ok({
            "success": True,
            "symbol": args["symbol"],
            "company_name": report.company_name,
            "risk_level": report.risk_level,
            "total_anomalies": report.total_anomalies,
            "high_severity_count": report.high_severity_count,
            "quality_scores": {
                "m_score": report.m_score,
                "z_score": report.z_score,
                "f_score": report.f_score,
                "sloan_ratio": report.sloan_ratio,
            },
            "anomalies": anomalies_summary,
        })
    except Exception as e:
        return _err(str(e))


@tool(
    "get_stock_fundamentals",
    "Get fundamental financial data for a specific stock from the loaded screener data.",
    {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "market": {"type": "string", "description": "'US' or 'SG'"},
        },
        "required": ["symbol"],
    },
)
async def get_stock_fundamentals(args: Dict[str, Any]) -> Dict[str, Any]:
    from data_loader import DataLoader

    try:
        loader = DataLoader()
        df = loader.load_screener_data(args.get("market", "US"))
        stock = df[df["Symbol"] == args["symbol"].upper()]

        if len(stock) == 0:
            return _err(f"Stock {args['symbol']} not found in {args.get('market', 'US')} market")

        return _ok({"success": True, "data": stock.iloc[0].to_dict()})
    except Exception as e:
        return _err(str(e))


@tool(
    "compare_stocks",
    "Compare multiple stocks side by side on key metrics.",
    {
        "type": "object",
        "properties": {
            "symbols": {"type": "array", "items": {"type": "string"}},
            "market": {"type": "string", "description": "'US' or 'SG'"},
        },
        "required": ["symbols"],
    },
)
async def compare_stocks(args: Dict[str, Any]) -> Dict[str, Any]:
    from data_loader import DataLoader

    try:
        loader = DataLoader()
        df = loader.load_screener_data(args.get("market", "US"))

        comparison = []
        for symbol in args["symbols"]:
            stock = df[df["Symbol"] == symbol.upper()]
            if len(stock) > 0:
                comparison.append(stock.iloc[0].to_dict())

        return _ok({
            "success": True,
            "stocks_found": len(comparison),
            "comparison": comparison,
        })
    except Exception as e:
        return _err(str(e))


@tool(
    "web_search",
    "Search the web for real-time information about stocks, companies, or market news (Tavily).",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "description": "Default 5"},
        },
        "required": ["query"],
    },
)
async def web_search(args: Dict[str, Any]) -> Dict[str, Any]:
    if not TAVILY_AVAILABLE:
        return _err("Tavily API key not configured. Set TAVILY_API_KEY environment variable.")

    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=args["query"],
            search_depth="advanced",
            max_results=args.get("max_results", 5),
            include_answer=True,
        )

        results = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content", "")[:500],
            }
            for r in response.get("results", [])
        ]

        return _ok({
            "success": True,
            "answer": response.get("answer", ""),
            "results": results,
        })
    except Exception as e:
        return _err(str(e))


@tool(
    "get_realtime_price",
    "Get real-time stock price and quote data from Twelve Data.",
    {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
        },
        "required": ["symbol"],
    },
)
async def get_realtime_price(args: Dict[str, Any]) -> Dict[str, Any]:
    twelve_data_key = os.getenv("TWELVE_DATA_API_KEY")
    if not twelve_data_key:
        return _err("Twelve Data API key not configured. Set TWELVE_DATA_API_KEY.")
    if not HTTPX_AVAILABLE:
        return _err("httpx not available")

    try:
        with httpx.Client() as client:
            response = client.get(
                "https://api.twelvedata.com/quote",
                params={"symbol": args["symbol"].upper(), "apikey": twelve_data_key},
                timeout=10.0,
            )
            data = response.json()

            if "code" in data:
                return _err(data.get("message", "Unknown error"))

            return _ok({
                "success": True,
                "symbol": data.get("symbol"),
                "name": data.get("name"),
                "exchange": data.get("exchange"),
                "price": data.get("close"),
                "open": data.get("open"),
                "high": data.get("high"),
                "low": data.get("low"),
                "volume": data.get("volume"),
                "change": data.get("change"),
                "percent_change": data.get("percent_change"),
                "previous_close": data.get("previous_close"),
                "fifty_two_week_high": data.get("fifty_two_week", {}).get("high"),
                "fifty_two_week_low": data.get("fifty_two_week", {}).get("low"),
                "timestamp": data.get("datetime"),
            })
    except Exception as e:
        return _err(str(e))


@tool(
    "get_price_history",
    "Get historical price data (OHLCV) for a stock from Twelve Data.",
    {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "interval": {
                "type": "string",
                "description": "1min, 5min, 15min, 30min, 1h, 1day, 1week, 1month",
            },
            "outputsize": {"type": "integer", "description": "Number of data points (max 100)"},
        },
        "required": ["symbol"],
    },
)
async def get_price_history(args: Dict[str, Any]) -> Dict[str, Any]:
    twelve_data_key = os.getenv("TWELVE_DATA_API_KEY")
    if not twelve_data_key:
        return _err("Twelve Data API key not configured. Set TWELVE_DATA_API_KEY.")
    if not HTTPX_AVAILABLE:
        return _err("httpx not available")

    try:
        with httpx.Client() as client:
            response = client.get(
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol": args["symbol"].upper(),
                    "interval": args.get("interval", "1day"),
                    "outputsize": min(args.get("outputsize", 30), 100),
                    "apikey": twelve_data_key,
                },
                timeout=15.0,
            )
            data = response.json()

            if "code" in data:
                return _err(data.get("message", "Unknown error"))

            values = data.get("values", [])
            formatted = [
                {
                    "date": v.get("datetime"),
                    "open": float(v.get("open", 0)),
                    "high": float(v.get("high", 0)),
                    "low": float(v.get("low", 0)),
                    "close": float(v.get("close", 0)),
                    "volume": int(v.get("volume", 0)),
                }
                for v in values[:30]
            ]

            return _ok({
                "success": True,
                "symbol": data.get("meta", {}).get("symbol"),
                "interval": args.get("interval", "1day"),
                "data_points": len(formatted),
                "prices": formatted,
            })
    except Exception as e:
        return _err(str(e))


@tool(
    "get_market_news",
    "Get latest market news and financial headlines (NewsAPI, falls back to Tavily).",
    {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "News topic (e.g., 'AAPL earnings')"},
            "count": {"type": "integer", "description": "Number of articles (default 5)"},
        },
        "required": ["topic"],
    },
)
async def get_market_news(args: Dict[str, Any]) -> Dict[str, Any]:
    if not HTTPX_AVAILABLE:
        return _err("httpx not available")

    topic = args["topic"]
    count = args.get("count", 5)
    news_api_key = os.getenv("NEWS_API_KEY")

    if news_api_key:
        try:
            with httpx.Client() as client:
                response = client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": topic,
                        "sortBy": "publishedAt",
                        "pageSize": count,
                        "apiKey": news_api_key,
                    },
                    timeout=10.0,
                )
                data = response.json()

                if data.get("status") == "ok":
                    articles = [
                        {
                            "title": a.get("title"),
                            "source": a.get("source", {}).get("name"),
                            "published": a.get("publishedAt"),
                            "description": a.get("description"),
                            "url": a.get("url"),
                        }
                        for a in data.get("articles", [])[:count]
                    ]
                    return _ok({"success": True, "articles": articles})
        except Exception:
            pass

    if TAVILY_AVAILABLE:
        return await web_search({"query": f"latest news {topic}", "max_results": count})

    return _err("No news API configured. Set NEWS_API_KEY or TAVILY_API_KEY.")


# ============================================================================
# MCP server packaging the tool functions
# ============================================================================

ALL_TOOLS = [
    screen_stocks,
    analyze_valuation,
    detect_anomalies,
    get_stock_fundamentals,
    compare_stocks,
    web_search,
    get_realtime_price,
    get_price_history,
    get_market_news,
]

VALUE_MCP_SERVER = create_sdk_mcp_server(
    name="value_tools",
    version="1.0.0",
    tools=ALL_TOOLS,
)

# Tool name format for allowed_tools: mcp__<server_name>__<tool_name>
TOOL_NAMES = {
    "screen_stocks": "mcp__value_tools__screen_stocks",
    "analyze_valuation": "mcp__value_tools__analyze_valuation",
    "detect_anomalies": "mcp__value_tools__detect_anomalies",
    "get_stock_fundamentals": "mcp__value_tools__get_stock_fundamentals",
    "compare_stocks": "mcp__value_tools__compare_stocks",
    "web_search": "mcp__value_tools__web_search",
    "get_realtime_price": "mcp__value_tools__get_realtime_price",
    "get_price_history": "mcp__value_tools__get_price_history",
    "get_market_news": "mcp__value_tools__get_market_news",
}

COMMON_TOOL_NAMES = [
    TOOL_NAMES["screen_stocks"],
    TOOL_NAMES["analyze_valuation"],
    TOOL_NAMES["detect_anomalies"],
    TOOL_NAMES["get_stock_fundamentals"],
    TOOL_NAMES["compare_stocks"],
]

WEB_TOOL_NAMES = [
    TOOL_NAMES["web_search"],
    TOOL_NAMES["get_market_news"],
    TOOL_NAMES["get_realtime_price"],
    TOOL_NAMES["get_price_history"],
]


# ============================================================================
# Subagent definitions and system prompts
# ============================================================================

SCREENING_SYSTEM_PROMPT = """You are a stock screening specialist. Your job is to help investors find stocks that match their criteria.

When screening stocks:
1. Ask clarifying questions about criteria if needed
2. Use appropriate thresholds based on value investing principles
3. Explain why certain criteria matter
4. Highlight the best matches and why they stand out

Default reasonable criteria for value stocks:
- Gross Margin > 20%
- ROE > 10%
- Debt-to-Equity < 1.5
- Positive FCF Margin
- ROIC > WACC (positive spread)

Use the screen_stocks tool to filter stocks. Use compare_stocks to compare top picks.
"""

ANOMALY_SYSTEM_PROMPT = """You are a forensic accounting and anomaly detection specialist. Your job is to analyze companies for financial red flags.

When analyzing:
1. Explain what each quality metric means (M-Score, Z-Score, F-Score, Sloan Ratio)
2. Distinguish between legitimate business reasons and concerning patterns
3. Provide specific investigation recommendations
4. Rate the overall risk level and explain why

Key thresholds to remember:
- M-Score > -1.78: Higher probability of earnings manipulation
- Z-Score < 1.8: Financial distress zone (bankruptcy risk)
- Z-Score 1.8-3.0: Grey zone
- Z-Score > 3.0: Safe zone
- F-Score < 3: Weak financial position
- F-Score > 6: Strong financial position
- Sloan Ratio > 10%: Earnings quality concerns (high accruals)

Use detect_anomalies tool for detailed analysis. Always explain findings in plain language.
"""

RESEARCH_SYSTEM_PROMPT = """You are an investment research analyst. Your job is to provide in-depth analysis and investment thesis for stocks.

When analyzing:
1. Evaluate both bull and bear cases objectively
2. Identify key metrics to monitor going forward
3. Assess margin of safety using EPV vs Market Cap
4. Provide clear buy/hold/avoid recommendations with reasoning
5. Search for recent news and developments that may impact the investment

Focus on value investing principles:
- Look for undervalued opportunities (EPV/MC > 1.3)
- Emphasize earnings power and sustainability
- Consider balance sheet strength
- Factor in competitive position and moat

Use web_search and get_market_news for real-time information.
Use get_stock_fundamentals for financial data.
"""

COORDINATOR_SYSTEM_PROMPT = """You are the Value Investment AI Coordinator. You help investors screen stocks, analyze valuations, and detect financial anomalies.

You coordinate a team of specialized agents available via the Task tool:
1. **screening**: Expert at filtering stocks based on fundamental criteria
2. **anomaly**: Expert at detecting financial red flags and forensic analysis
3. **research**: Expert at investment research with real-time web data

Route requests appropriately:
- Stock screening, filtering, comparisons → delegate to 'screening' subagent
- M-Score, Z-Score, anomaly detection, red flags → delegate to 'anomaly' subagent
- Investment thesis, news, real-time data, buy/sell recommendations → delegate to 'research' subagent

For simple queries, you can answer directly using your tools.
For complex tasks, delegate to the appropriate specialist via Task.

Always be data-driven and cite specific metrics when making recommendations.
Focus on value investing principles: margin of safety, earnings power, and financial strength.
"""


def _subagent_definitions() -> Dict[str, AgentDefinition]:
    """Define screening/anomaly/research subagents for the coordinator."""
    return {
        "screening": AgentDefinition(
            description="Stock screening specialist for filtering by fundamental criteria",
            prompt=SCREENING_SYSTEM_PROMPT,
            tools=COMMON_TOOL_NAMES,
            model="sonnet",
        ),
        "anomaly": AgentDefinition(
            description="Forensic anomaly detection specialist for financial red flags",
            prompt=ANOMALY_SYSTEM_PROMPT,
            tools=COMMON_TOOL_NAMES,
            model="sonnet",
        ),
        "research": AgentDefinition(
            description="Investment research analyst with real-time web access",
            prompt=RESEARCH_SYSTEM_PROMPT,
            tools=COMMON_TOOL_NAMES + WEB_TOOL_NAMES,
            model="sonnet",
        ),
    }


# ============================================================================
# Agent Response Wrapper for backward compatibility with app.py
# ============================================================================

@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    tool_calls: List[Dict] = field(default_factory=list)
    raw_response: Any = None


class ValueInvestmentAgent:
    """
    Wrapper class for the Value Investment Agent system.
    Backward-compatible interface; uses Claude Agent SDK underneath.
    """

    def __init__(self, agent_type: str = "coordinator", model: str = DEFAULT_MODEL):
        """
        Args:
            agent_type: 'coordinator', 'screening', 'anomaly', or 'research'
            model: Claude model ID. Default is claude-sonnet-4-6.
        """
        self.agent_type = agent_type
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self._options = self._build_options()

    def _build_options(self) -> ClaudeAgentOptions:
        if self.agent_type == "coordinator":
            return ClaudeAgentOptions(
                system_prompt=COORDINATOR_SYSTEM_PROMPT,
                model=self.model,
                mcp_servers={"value_tools": VALUE_MCP_SERVER},
                agents=_subagent_definitions(),
                allowed_tools=COMMON_TOOL_NAMES + WEB_TOOL_NAMES + ["Task"],
                permission_mode="bypassPermissions",
            )
        if self.agent_type == "screening":
            return ClaudeAgentOptions(
                system_prompt=SCREENING_SYSTEM_PROMPT,
                model=self.model,
                mcp_servers={"value_tools": VALUE_MCP_SERVER},
                allowed_tools=COMMON_TOOL_NAMES,
                permission_mode="bypassPermissions",
            )
        if self.agent_type == "anomaly":
            return ClaudeAgentOptions(
                system_prompt=ANOMALY_SYSTEM_PROMPT,
                model=self.model,
                mcp_servers={"value_tools": VALUE_MCP_SERVER},
                allowed_tools=COMMON_TOOL_NAMES,
                permission_mode="bypassPermissions",
            )
        if self.agent_type == "research":
            return ClaudeAgentOptions(
                system_prompt=RESEARCH_SYSTEM_PROMPT,
                model=self.model,
                mcp_servers={"value_tools": VALUE_MCP_SERVER},
                allowed_tools=COMMON_TOOL_NAMES + WEB_TOOL_NAMES,
                permission_mode="bypassPermissions",
            )
        # Fallback
        return ClaudeAgentOptions(
            system_prompt=COORDINATOR_SYSTEM_PROMPT,
            model=self.model,
            mcp_servers={"value_tools": VALUE_MCP_SERVER},
            agents=_subagent_definitions(),
            allowed_tools=COMMON_TOOL_NAMES + WEB_TOOL_NAMES + ["Task"],
            permission_mode="bypassPermissions",
        )

    def chat(self, user_message: str) -> AgentResponse:
        """Sync wrapper around chat_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an existing event loop (e.g., Streamlit nested) — run in a task.
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.chat_async(user_message))
            return loop.run_until_complete(self.chat_async(user_message))
        except RuntimeError:
            return asyncio.run(self.chat_async(user_message))

    async def chat_async(self, user_message: str) -> AgentResponse:
        """Send a message and collect the assistant response."""
        self.conversation_history.append({"role": "user", "content": user_message})

        text_chunks: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        last_msg = None

        try:
            async with ClaudeSDKClient(options=self._options) as client:
                await client.query(user_message)
                async for msg in client.receive_response():
                    last_msg = msg
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                text_chunks.append(block.text)
                            elif isinstance(block, ToolUseBlock):
                                tool_calls.append({
                                    "tool": block.name,
                                    "input": block.input,
                                })
                    elif isinstance(msg, ResultMessage):
                        break

            final_text = "".join(text_chunks).strip() or "(no response)"
            self.conversation_history.append({"role": "assistant", "content": final_text})

            return AgentResponse(
                content=final_text,
                tool_calls=tool_calls,
                raw_response=last_msg,
            )
        except Exception as e:
            return AgentResponse(content=f"Agent error: {str(e)}")

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []


# Convenience aliases for backward compatibility
class ScreeningAgent(ValueInvestmentAgent):
    """Specialized agent for stock screening."""
    def __init__(self):
        super().__init__(agent_type="screening")


class AnomalyAgent(ValueInvestmentAgent):
    """Specialized agent for anomaly detection."""
    def __init__(self):
        super().__init__(agent_type="anomaly")


class ResearchAgent(ValueInvestmentAgent):
    """Specialized agent for investment research."""
    def __init__(self):
        super().__init__(agent_type="research")


# ============================================================================
# Convenience Functions
# ============================================================================

def get_agent(agent_type: str = "coordinator") -> ValueInvestmentAgent:
    """Get an agent instance by type."""
    return ValueInvestmentAgent(agent_type=agent_type)


async def quick_screen_async(market: str, **criteria) -> str:
    """Quick screening with AI analysis (async)."""
    agent = ScreeningAgent()
    criteria_str = ", ".join(f"{k}={v}" for k, v in criteria.items())
    prompt = f"Screen {market} stocks with these criteria: {criteria_str}. Analyze the results and recommend top picks."
    response = await agent.chat_async(prompt)
    return response.content


def quick_screen(market: str, **criteria) -> str:
    """Quick screening with AI analysis (sync wrapper)."""
    return asyncio.run(quick_screen_async(market, **criteria))


async def quick_analyze_async(symbol: str) -> str:
    """Quick anomaly analysis with AI interpretation (async)."""
    agent = AnomalyAgent()
    prompt = f"Analyze {symbol} for financial anomalies and red flags. Provide a detailed risk assessment."
    response = await agent.chat_async(prompt)
    return response.content


def quick_analyze(symbol: str) -> str:
    """Quick anomaly analysis (sync wrapper)."""
    return asyncio.run(quick_analyze_async(symbol))


async def run_full_analysis(symbol: str, market: str = "US") -> Dict[str, Any]:
    """
    Run a full sequential analysis: Screening fundamentals → Anomaly → Research thesis.

    Args:
        symbol: Stock ticker symbol
        market: Market (US or SG)

    Returns:
        Dictionary with results from all agents
    """
    results: Dict[str, Any] = {}

    screening_agent = ScreeningAgent()
    fundamentals = await screening_agent.chat_async(
        f"Get the fundamentals for {symbol} in the {market} market and evaluate its valuation."
    )
    results["fundamentals"] = fundamentals.content

    anomaly_agent = AnomalyAgent()
    anomaly = await anomaly_agent.chat_async(
        f"Analyze {symbol} for any financial red flags or anomalies. Explain the risk level."
    )
    results["anomalies"] = anomaly.content

    research_agent = ResearchAgent()
    research = await research_agent.chat_async(
        f"""Based on this analysis for {symbol}:

Fundamentals: {fundamentals.content[:500]}
Anomalies: {anomaly.content[:500]}

Search for recent news about {symbol} and provide a complete investment thesis with recommendation."""
    )
    results["thesis"] = research.content

    return results


# ============================================================================
# Demo / Smoke Test
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("=" * 60)
        print("Value Investment AI Agent Demo (Claude Agent SDK)")
        print("=" * 60)

        agent = ValueInvestmentAgent()

        print("\n[Test 1: Stock Screening]")
        response = await agent.chat_async("Screen US stocks with ROE > 15% and gross margin > 30%")
        print(f"Response: {response.content[:500]}...")

        print("\n[Test 2: Anomaly Detection]")
        agent.reset_conversation()
        response = await agent.chat_async("Analyze DDI for any financial anomalies")
        print(f"Response: {response.content[:500]}...")

        print("\n" + "=" * 60)
        print("Demo complete!")

    asyncio.run(demo())
