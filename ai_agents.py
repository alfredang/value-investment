"""
AI Agents for Value Investment Analysis using OpenAI Agents SDK.

This module provides specialized agents with:
- Handoff patterns for agent coordination
- Web search via Tavily for real-time data
- News API for market updates
- Tool-based stock screening and anomaly detection
"""
import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Agents SDK imports
from agents import Agent, Runner, function_tool, handoff, RunContextWrapper
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

# Optional: Tavily for web search
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = bool(os.getenv("TAVILY_API_KEY"))
except ImportError:
    TAVILY_AVAILABLE = False

# Optional: httpx for news API
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# ============================================================================
# Tool Functions - Decorated with @function_tool for Agents SDK
# ============================================================================

@function_tool
def screen_stocks(
    market: str,
    gross_margin_min: float = None,
    net_margin_min: float = None,
    roe_min: float = None,
    roa_min: float = None,
    debt_equity_max: float = None,
    fcf_margin_min: float = None,
    roic_wacc_min: float = None
) -> str:
    """
    Screen stocks based on fundamental criteria.

    Args:
        market: 'US' or 'SG' market
        gross_margin_min: Minimum gross margin percentage
        net_margin_min: Minimum net margin percentage
        roe_min: Minimum return on equity
        roa_min: Minimum return on assets
        debt_equity_max: Maximum debt-to-equity ratio
        fcf_margin_min: Minimum free cash flow margin
        roic_wacc_min: Minimum ROIC minus WACC

    Returns:
        JSON string with screening results
    """
    from screener import StockScreener
    from data_loader import DataLoader

    try:
        loader = DataLoader()
        screener = StockScreener(loader)

        criteria = {}
        if gross_margin_min is not None:
            criteria['gross_margin'] = gross_margin_min
        if net_margin_min is not None:
            criteria['net_margin'] = net_margin_min
        if roe_min is not None:
            criteria['roe'] = roe_min
        if roa_min is not None:
            criteria['roa'] = roa_min
        if debt_equity_max is not None:
            criteria['debt_to_equity'] = debt_equity_max
        if fcf_margin_min is not None:
            criteria['fcf_margin'] = fcf_margin_min
        if roic_wacc_min is not None:
            criteria['roic_wacc'] = roic_wacc_min

        results = screener.screen(market=market, criteria=criteria)

        return json.dumps({
            "success": True,
            "total_matches": len(results),
            "criteria_used": criteria,
            "top_stocks": results.head(20).to_dict('records'),
            "valuation_summary": results['Valuation'].value_counts().to_dict() if 'Valuation' in results.columns else {}
        }, default=str)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@function_tool
def analyze_valuation(symbol: str, epv: float, market_cap: float) -> str:
    """
    Analyze stock valuation using EPV vs Market Cap.

    Args:
        symbol: Stock ticker symbol
        epv: Earnings Power Value
        market_cap: Market capitalization in millions

    Returns:
        JSON string with valuation analysis
    """
    from valuation import Valuator, ValuationStatus

    valuator = Valuator()
    status, ratio, margin = valuator.classify(epv, market_cap)

    interpretation = ""
    if status and ratio:
        if status.value == "Undervalued":
            interpretation = f"Stock appears undervalued with {margin:.1f}% margin of safety."
        elif status.value == "Fair Value":
            interpretation = f"Stock is fairly valued. EPV is approximately {ratio:.1%} of market cap."
        elif status.value == "Overvalued":
            interpretation = f"Stock appears overvalued. EPV is only {ratio:.1%} of market cap."

    return json.dumps({
        "symbol": symbol,
        "epv": epv,
        "market_cap": market_cap,
        "epv_mc_ratio": ratio,
        "valuation_status": status.value if status else "N/A",
        "margin_of_safety_pct": margin,
        "interpretation": interpretation
    }, default=str)


@function_tool
def detect_anomalies(symbol: str) -> str:
    """
    Detect financial anomalies in a company's financials using M-Score, Z-Score, F-Score.

    Args:
        symbol: Stock ticker symbol (e.g., DDI, TEX, CHCI)

    Returns:
        JSON string with anomaly detection report
    """
    from anomaly_detector import AnomalyDetector
    from data_loader import DataLoader

    try:
        loader = DataLoader()
        detector = AnomalyDetector(loader)
        report = detector.analyze(symbol)

        anomalies_summary = []
        for a in report.anomalies[:15]:
            anomalies_summary.append({
                "category": a.category,
                "severity": a.severity.value,
                "description": a.description,
                "year": a.year,
                "details": a.details
            })

        return json.dumps({
            "success": True,
            "symbol": symbol,
            "company_name": report.company_name,
            "risk_level": report.risk_level,
            "total_anomalies": report.total_anomalies,
            "high_severity_count": report.high_severity_count,
            "quality_scores": {
                "m_score": report.m_score,
                "z_score": report.z_score,
                "f_score": report.f_score,
                "sloan_ratio": report.sloan_ratio
            },
            "anomalies": anomalies_summary
        }, default=str)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@function_tool
def get_stock_fundamentals(symbol: str, market: str = "US") -> str:
    """
    Get fundamental financial data for a specific stock.

    Args:
        symbol: Stock ticker symbol
        market: 'US' or 'SG' market

    Returns:
        JSON string with stock fundamental data
    """
    from data_loader import DataLoader

    try:
        loader = DataLoader()
        df = loader.load_screener_data(market)
        stock = df[df['Symbol'] == symbol.upper()]

        if len(stock) == 0:
            return json.dumps({"success": False, "error": f"Stock {symbol} not found in {market} market"})

        return json.dumps({
            "success": True,
            "data": stock.iloc[0].to_dict()
        }, default=str)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@function_tool
def compare_stocks(symbols: List[str], market: str = "US") -> str:
    """
    Compare multiple stocks side by side on key metrics.

    Args:
        symbols: List of stock ticker symbols to compare
        market: 'US' or 'SG' market

    Returns:
        JSON string with comparison data
    """
    from data_loader import DataLoader

    try:
        loader = DataLoader()
        df = loader.load_screener_data(market)

        comparison = []
        for symbol in symbols:
            stock = df[df['Symbol'] == symbol.upper()]
            if len(stock) > 0:
                comparison.append(stock.iloc[0].to_dict())

        return json.dumps({
            "success": True,
            "stocks_found": len(comparison),
            "comparison": comparison
        }, default=str)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@function_tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for real-time information about stocks, companies, or market news.

    Args:
        query: Search query (e.g., "AAPL latest earnings report")
        max_results: Maximum number of results to return

    Returns:
        JSON string with search results
    """
    if not TAVILY_AVAILABLE:
        return json.dumps({
            "success": False,
            "error": "Tavily API key not configured. Set TAVILY_API_KEY environment variable."
        })

    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content", "")[:500]
            })

        return json.dumps({
            "success": True,
            "answer": response.get("answer", ""),
            "results": results
        }, default=str)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@function_tool
def get_realtime_price(symbol: str) -> str:
    """
    Get real-time stock price and quote data from Twelve Data.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL", "MSFT")

    Returns:
        JSON string with current price, change, volume
    """
    twelve_data_key = os.getenv("TWELVE_DATA_API_KEY")

    if not twelve_data_key:
        return json.dumps({
            "success": False,
            "error": "Twelve Data API key not configured. Set TWELVE_DATA_API_KEY."
        })

    if not HTTPX_AVAILABLE:
        return json.dumps({"success": False, "error": "httpx not available"})

    try:
        with httpx.Client() as client:
            # Get quote data
            response = client.get(
                "https://api.twelvedata.com/quote",
                params={
                    "symbol": symbol.upper(),
                    "apikey": twelve_data_key
                },
                timeout=10.0
            )
            data = response.json()

            if "code" in data:  # Error response
                return json.dumps({"success": False, "error": data.get("message", "Unknown error")})

            return json.dumps({
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
                "timestamp": data.get("datetime")
            }, default=str)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@function_tool
def get_price_history(symbol: str, interval: str = "1day", outputsize: int = 30) -> str:
    """
    Get historical price data for a stock.

    Args:
        symbol: Stock ticker symbol (e.g., "AAPL")
        interval: Time interval (1min, 5min, 15min, 30min, 1h, 1day, 1week, 1month)
        outputsize: Number of data points to return (max 5000)

    Returns:
        JSON string with historical price data
    """
    twelve_data_key = os.getenv("TWELVE_DATA_API_KEY")

    if not twelve_data_key:
        return json.dumps({
            "success": False,
            "error": "Twelve Data API key not configured. Set TWELVE_DATA_API_KEY."
        })

    if not HTTPX_AVAILABLE:
        return json.dumps({"success": False, "error": "httpx not available"})

    try:
        with httpx.Client() as client:
            response = client.get(
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "outputsize": min(outputsize, 100),  # Limit for free tier
                    "apikey": twelve_data_key
                },
                timeout=15.0
            )
            data = response.json()

            if "code" in data:
                return json.dumps({"success": False, "error": data.get("message", "Unknown error")})

            # Format the time series data
            values = data.get("values", [])
            formatted_data = []
            for v in values[:30]:  # Limit to recent data
                formatted_data.append({
                    "date": v.get("datetime"),
                    "open": float(v.get("open", 0)),
                    "high": float(v.get("high", 0)),
                    "low": float(v.get("low", 0)),
                    "close": float(v.get("close", 0)),
                    "volume": int(v.get("volume", 0))
                })

            return json.dumps({
                "success": True,
                "symbol": data.get("meta", {}).get("symbol"),
                "interval": interval,
                "data_points": len(formatted_data),
                "prices": formatted_data
            }, default=str)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@function_tool
def get_market_news(topic: str = "stock market", count: int = 5) -> str:
    """
    Get latest market news and financial headlines.

    Args:
        topic: News topic (e.g., "stock market", "earnings", company name)
        count: Number of news items to retrieve

    Returns:
        JSON string with news articles
    """
    if not HTTPX_AVAILABLE:
        return json.dumps({"success": False, "error": "httpx not available"})

    # Use free NewsAPI or fallback to web search
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
                        "apiKey": news_api_key
                    },
                    timeout=10.0
                )
                data = response.json()

                if data.get("status") == "ok":
                    articles = []
                    for article in data.get("articles", [])[:count]:
                        articles.append({
                            "title": article.get("title"),
                            "source": article.get("source", {}).get("name"),
                            "published": article.get("publishedAt"),
                            "description": article.get("description"),
                            "url": article.get("url")
                        })
                    return json.dumps({"success": True, "articles": articles})
        except Exception as e:
            pass  # Fall through to web search

    # Fallback to Tavily if available
    if TAVILY_AVAILABLE:
        return web_search(f"latest news {topic}", max_results=count)

    return json.dumps({
        "success": False,
        "error": "No news API configured. Set NEWS_API_KEY or TAVILY_API_KEY."
    })


# ============================================================================
# Agent Definitions using OpenAI Agents SDK
# ============================================================================

# Common tools for all agents
COMMON_TOOLS = [
    screen_stocks,
    analyze_valuation,
    detect_anomalies,
    get_stock_fundamentals,
    compare_stocks,
]

# Web-enabled tools
WEB_TOOLS = [
    web_search,
    get_market_news,
    get_realtime_price,
    get_price_history,
]


def create_screening_agent() -> Agent:
    """Create the stock screening specialist agent."""
    return Agent(
        name="ScreeningAgent",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are a stock screening specialist. Your job is to help investors find stocks that match their criteria.

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
If the user needs anomaly detection, hand off to the AnomalyAgent.
If the user needs investment research, hand off to the ResearchAgent.
""",
        tools=COMMON_TOOLS,
    )


def create_anomaly_agent() -> Agent:
    """Create the forensic accounting/anomaly detection agent."""
    return Agent(
        name="AnomalyAgent",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are a forensic accounting and anomaly detection specialist. Your job is to analyze companies for financial red flags.

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
If user needs stock screening, hand off to the ScreeningAgent.
If user needs investment thesis, hand off to the ResearchAgent.
""",
        tools=COMMON_TOOLS,
    )


def create_research_agent() -> Agent:
    """Create the investment research agent with web access."""
    return Agent(
        name="ResearchAgent",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are an investment research analyst. Your job is to provide in-depth analysis and investment thesis for stocks.

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
If detailed screening needed, hand off to ScreeningAgent.
If anomaly analysis needed, hand off to AnomalyAgent.
""",
        tools=COMMON_TOOLS + WEB_TOOLS,
    )


def create_coordinator_agent() -> Agent:
    """Create the main coordinator agent that manages handoffs."""

    # Create sub-agents
    screening_agent = create_screening_agent()
    anomaly_agent = create_anomaly_agent()
    research_agent = create_research_agent()

    # Create handoffs
    handoff_to_screening = handoff(
        agent=screening_agent,
        tool_name_override="transfer_to_screening_agent",
        tool_description_override="Transfer to the Screening Agent for stock filtering and screening tasks."
    )

    handoff_to_anomaly = handoff(
        agent=anomaly_agent,
        tool_name_override="transfer_to_anomaly_agent",
        tool_description_override="Transfer to the Anomaly Agent for forensic analysis and red flag detection."
    )

    handoff_to_research = handoff(
        agent=research_agent,
        tool_name_override="transfer_to_research_agent",
        tool_description_override="Transfer to the Research Agent for investment thesis and real-time market data."
    )

    return Agent(
        name="ValueInvestmentCoordinator",
        instructions=f"""{RECOMMENDED_PROMPT_PREFIX}

You are the Value Investment AI Coordinator. You help investors screen stocks, analyze valuations, and detect financial anomalies.

You coordinate a team of specialized agents:
1. **ScreeningAgent**: Expert at filtering stocks based on fundamental criteria
2. **AnomalyAgent**: Expert at detecting financial red flags and forensic analysis
3. **ResearchAgent**: Expert at investment research with real-time web data

Route requests appropriately:
- Stock screening, filtering, comparisons → ScreeningAgent
- M-Score, Z-Score, anomaly detection, red flags → AnomalyAgent
- Investment thesis, news, real-time data, buy/sell recommendations → ResearchAgent

For simple queries, you can answer directly using your tools.
For complex tasks, delegate to the appropriate specialist.

Always be data-driven and cite specific metrics when making recommendations.
Focus on value investing principles: margin of safety, earnings power, and financial strength.
""",
        tools=COMMON_TOOLS + [handoff_to_screening, handoff_to_anomaly, handoff_to_research],
    )


# ============================================================================
# Agent Response Wrapper for Compatibility
# ============================================================================

@dataclass
class AgentResponse:
    """Response from an agent - compatible with existing code."""
    content: str
    tool_calls: List[Dict] = field(default_factory=list)
    raw_response: Any = None


class ValueInvestmentAgent:
    """
    Wrapper class for the Value Investment Agent system.
    Provides backward-compatible interface while using Agents SDK.
    """

    def __init__(self, agent_type: str = "coordinator"):
        """
        Initialize the agent.

        Args:
            agent_type: Type of agent ('coordinator', 'screening', 'anomaly', 'research')
        """
        self.agent_type = agent_type

        if agent_type == "coordinator":
            self.agent = create_coordinator_agent()
        elif agent_type == "screening":
            self.agent = create_screening_agent()
        elif agent_type == "anomaly":
            self.agent = create_anomaly_agent()
        elif agent_type == "research":
            self.agent = create_research_agent()
        else:
            self.agent = create_coordinator_agent()

        self.conversation_history = []

    def chat(self, user_message: str) -> AgentResponse:
        """
        Chat with the agent synchronously.

        Args:
            user_message: User's message/question

        Returns:
            AgentResponse with the agent's response
        """
        # Run async in sync context
        return asyncio.get_event_loop().run_until_complete(
            self.chat_async(user_message)
        )

    async def chat_async(self, user_message: str) -> AgentResponse:
        """
        Chat with the agent asynchronously.

        Args:
            user_message: User's message/question

        Returns:
            AgentResponse with the agent's response
        """
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            # Run the agent
            result = await Runner.run(
                self.agent,
                input=user_message,
            )

            # Extract tool calls from the run
            tool_calls = []
            if hasattr(result, 'new_items'):
                for item in result.new_items:
                    if hasattr(item, 'raw_item') and hasattr(item.raw_item, 'type'):
                        if item.raw_item.type == 'function_call':
                            tool_calls.append({
                                "tool": item.raw_item.name if hasattr(item.raw_item, 'name') else "unknown",
                                "arguments": item.raw_item.arguments if hasattr(item.raw_item, 'arguments') else {}
                            })

            # Get the final output
            final_output = result.final_output if hasattr(result, 'final_output') else str(result)

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": final_output
            })

            return AgentResponse(
                content=final_output,
                tool_calls=tool_calls,
                raw_response=result
            )

        except Exception as e:
            error_msg = f"Agent error: {str(e)}"
            return AgentResponse(content=error_msg)

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
    """
    Get an agent instance.

    Args:
        agent_type: Type of agent ('coordinator', 'screening', 'anomaly', 'research')

    Returns:
        ValueInvestmentAgent instance
    """
    return ValueInvestmentAgent(agent_type=agent_type)


async def quick_screen_async(market: str, **criteria) -> str:
    """
    Quick screening with AI analysis (async).

    Args:
        market: 'US' or 'SG'
        **criteria: Screening criteria

    Returns:
        AI analysis of screening results
    """
    agent = ScreeningAgent()
    criteria_str = ", ".join(f"{k}={v}" for k, v in criteria.items())
    prompt = f"Screen {market} stocks with these criteria: {criteria_str}. Analyze the results and recommend top picks."
    response = await agent.chat_async(prompt)
    return response.content


def quick_screen(market: str, **criteria) -> str:
    """Quick screening with AI analysis (sync wrapper)."""
    return asyncio.get_event_loop().run_until_complete(
        quick_screen_async(market, **criteria)
    )


async def quick_analyze_async(symbol: str) -> str:
    """
    Quick anomaly analysis with AI interpretation (async).

    Args:
        symbol: Stock ticker symbol

    Returns:
        AI analysis of anomalies
    """
    agent = AnomalyAgent()
    prompt = f"Analyze {symbol} for financial anomalies and red flags. Provide a detailed risk assessment."
    response = await agent.chat_async(prompt)
    return response.content


def quick_analyze(symbol: str) -> str:
    """Quick anomaly analysis (sync wrapper)."""
    return asyncio.get_event_loop().run_until_complete(
        quick_analyze_async(symbol)
    )


# ============================================================================
# Sequential Agent Flow
# ============================================================================

async def run_full_analysis(symbol: str, market: str = "US") -> Dict[str, Any]:
    """
    Run a full sequential analysis: Screen → Anomaly → Research.

    Args:
        symbol: Stock ticker symbol
        market: Market (US or SG)

    Returns:
        Dictionary with results from all agents
    """
    results = {}

    # Step 1: Get fundamentals with screening agent
    screening_agent = ScreeningAgent()
    fundamentals_response = await screening_agent.chat_async(
        f"Get the fundamentals for {symbol} in the {market} market and evaluate its valuation."
    )
    results["fundamentals"] = fundamentals_response.content

    # Step 2: Run anomaly detection
    anomaly_agent = AnomalyAgent()
    anomaly_response = await anomaly_agent.chat_async(
        f"Analyze {symbol} for any financial red flags or anomalies. Explain the risk level."
    )
    results["anomalies"] = anomaly_response.content

    # Step 3: Generate investment thesis with research agent
    research_agent = ResearchAgent()
    research_response = await research_agent.chat_async(
        f"""Based on this analysis for {symbol}:

Fundamentals: {fundamentals_response.content[:500]}
Anomalies: {anomaly_response.content[:500]}

Search for recent news about {symbol} and provide a complete investment thesis with recommendation."""
    )
    results["thesis"] = research_response.content

    return results


# ============================================================================
# Main / Demo
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 60)
        print("Value Investment AI Agent Demo (OpenAI Agents SDK)")
        print("=" * 60)

        # Test with coordinator agent
        agent = ValueInvestmentAgent()

        # Test screening
        print("\n[Test 1: Stock Screening]")
        response = await agent.chat_async("Screen US stocks with ROE > 15% and gross margin > 30%")
        print(f"Response: {response.content[:500]}...")

        # Test anomaly detection
        print("\n[Test 2: Anomaly Detection]")
        agent.reset_conversation()
        response = await agent.chat_async("Analyze DDI for any financial anomalies")
        print(f"Response: {response.content[:500]}...")

        print("\n" + "=" * 60)
        print("Demo complete!")

    asyncio.run(demo())
