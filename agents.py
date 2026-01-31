"""
AI Agents for Value Investment Analysis using OpenAI Agents SDK.

This module provides specialized agents for:
- Stock screening and filtering
- Valuation analysis
- Anomaly detection and forensic analysis
- Investment research
"""
import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# Tool Functions - These are the tools that agents can use
# ============================================================================

def screen_stocks(
    market: str,
    gross_margin_min: float = None,
    net_margin_min: float = None,
    roe_min: float = None,
    roa_min: float = None,
    debt_equity_max: float = None,
    fcf_margin_min: float = None,
    roic_wacc_min: float = None
) -> Dict:
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
        Dictionary with screening results
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

        return {
            "success": True,
            "total_matches": len(results),
            "criteria_used": criteria,
            "top_stocks": results.head(20).to_dict('records'),
            "valuation_summary": results['Valuation'].value_counts().to_dict() if 'Valuation' in results.columns else {}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_valuation(symbol: str, epv: float, market_cap: float) -> Dict:
    """
    Analyze stock valuation using EPV vs Market Cap.

    Args:
        symbol: Stock ticker symbol
        epv: Earnings Power Value
        market_cap: Market capitalization

    Returns:
        Valuation analysis result
    """
    from valuation import Valuator, ValuationStatus

    valuator = Valuator()
    status, ratio, margin = valuator.classify(epv, market_cap)

    return {
        "symbol": symbol,
        "epv": epv,
        "market_cap": market_cap,
        "epv_mc_ratio": ratio,
        "valuation_status": status.value if status else "N/A",
        "margin_of_safety_pct": margin,
        "interpretation": _interpret_valuation(status, ratio, margin)
    }


def _interpret_valuation(status, ratio, margin):
    """Generate human-readable interpretation of valuation."""
    if status is None or ratio is None:
        return "Unable to determine valuation due to missing data."

    if status.value == "Undervalued":
        return f"Stock appears undervalued with {margin:.1f}% margin of safety. EPV suggests intrinsic value is {ratio:.1%} of market cap."
    elif status.value == "Fair Value":
        return f"Stock is fairly valued. EPV is approximately {ratio:.1%} of market cap."
    elif status.value == "Overvalued":
        return f"Stock appears overvalued. EPV is only {ratio:.1%} of market cap, suggesting {-margin:.1f}% downside."
    else:
        return "Valuation could not be determined."


def detect_anomalies(symbol: str) -> Dict:
    """
    Detect financial anomalies in a company's financials.

    Args:
        symbol: Stock ticker symbol (DDI, TEX, or CHCI)

    Returns:
        Anomaly detection report
    """
    from anomaly_detector import AnomalyDetector
    from data_loader import DataLoader

    try:
        loader = DataLoader()
        detector = AnomalyDetector(loader)
        report = detector.analyze(symbol)

        anomalies_summary = []
        for a in report.anomalies[:15]:  # Limit for API
            anomalies_summary.append({
                "category": a.category,
                "severity": a.severity.value,
                "description": a.description,
                "year": a.year,
                "details": a.details
            })

        return {
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
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_stock_fundamentals(symbol: str, market: str = "US") -> Dict:
    """
    Get fundamental data for a specific stock.

    Args:
        symbol: Stock ticker symbol
        market: 'US' or 'SG' market

    Returns:
        Stock fundamental data
    """
    from data_loader import DataLoader

    try:
        loader = DataLoader()
        df = loader.load_screener_data(market)
        stock = df[df['Symbol'] == symbol.upper()]

        if len(stock) == 0:
            return {"success": False, "error": f"Stock {symbol} not found in {market} market"}

        return {
            "success": True,
            "data": stock.iloc[0].to_dict()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def compare_stocks(symbols: List[str], market: str = "US") -> Dict:
    """
    Compare multiple stocks side by side.

    Args:
        symbols: List of stock ticker symbols
        market: 'US' or 'SG' market

    Returns:
        Comparison data
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

        return {
            "success": True,
            "stocks_found": len(comparison),
            "comparison": comparison
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# Tool Definitions for OpenAI Function Calling
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "screen_stocks",
            "description": "Screen stocks based on fundamental criteria like gross margin, ROE, debt-to-equity. Returns matching stocks with valuation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "market": {
                        "type": "string",
                        "enum": ["US", "SG"],
                        "description": "Market to screen (US or Singapore)"
                    },
                    "gross_margin_min": {
                        "type": "number",
                        "description": "Minimum gross margin percentage"
                    },
                    "net_margin_min": {
                        "type": "number",
                        "description": "Minimum net margin percentage"
                    },
                    "roe_min": {
                        "type": "number",
                        "description": "Minimum return on equity percentage"
                    },
                    "roa_min": {
                        "type": "number",
                        "description": "Minimum return on assets percentage"
                    },
                    "debt_equity_max": {
                        "type": "number",
                        "description": "Maximum debt-to-equity ratio"
                    },
                    "fcf_margin_min": {
                        "type": "number",
                        "description": "Minimum free cash flow margin percentage"
                    },
                    "roic_wacc_min": {
                        "type": "number",
                        "description": "Minimum ROIC minus WACC"
                    }
                },
                "required": ["market"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_valuation",
            "description": "Analyze a stock's valuation using EPV vs Market Cap to determine if undervalued, fair value, or overvalued",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "epv": {"type": "number", "description": "Earnings Power Value"},
                    "market_cap": {"type": "number", "description": "Market capitalization in millions"}
                },
                "required": ["symbol", "epv", "market_cap"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detect financial anomalies and red flags in a company's financials using M-Score, Z-Score, and other metrics",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., DDI, TEX, CHCI)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_fundamentals",
            "description": "Get fundamental financial data for a specific stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "market": {
                        "type": "string",
                        "enum": ["US", "SG"],
                        "description": "Market (US or Singapore)"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_stocks",
            "description": "Compare multiple stocks side by side",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stock ticker symbols to compare"
                    },
                    "market": {
                        "type": "string",
                        "enum": ["US", "SG"],
                        "description": "Market (US or Singapore)"
                    }
                },
                "required": ["symbols"]
            }
        }
    }
]


# ============================================================================
# Agent Classes
# ============================================================================

@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    tool_calls: List[Dict] = None
    raw_response: Any = None


class ValueInvestmentAgent:
    """
    Main AI agent for value investment analysis.

    This agent can:
    - Screen stocks based on criteria
    - Analyze valuations
    - Detect anomalies
    - Provide investment insights
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.system_prompt = """You are an expert value investment analyst AI assistant.
You help investors screen stocks, analyze valuations, and detect financial anomalies.

Your capabilities:
1. Screen stocks based on fundamental criteria (margins, ROE, debt levels, etc.)
2. Analyze valuations using EPV vs Market Cap methodology
3. Detect financial anomalies using M-Score, Z-Score, F-Score
4. Provide investment insights and recommendations

Always be data-driven and cite specific metrics when making recommendations.
Focus on value investing principles: margin of safety, earnings power, and financial strength.
"""
        self.conversation_history = []

    def _execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """Execute a tool and return the result as JSON string."""
        tool_functions = {
            "screen_stocks": screen_stocks,
            "analyze_valuation": analyze_valuation,
            "detect_anomalies": detect_anomalies,
            "get_stock_fundamentals": get_stock_fundamentals,
            "compare_stocks": compare_stocks
        }

        if tool_name in tool_functions:
            result = tool_functions[tool_name](**arguments)
            return json.dumps(result, default=str)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def chat(self, user_message: str) -> AgentResponse:
        """
        Chat with the agent.

        Args:
            user_message: User's message/question

        Returns:
            AgentResponse with the agent's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history

        # Call OpenAI
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        # Handle tool calls
        if assistant_message.tool_calls:
            # Add assistant message with tool calls
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            # Execute each tool
            tool_results = []
            for tool_call in assistant_message.tool_calls:
                arguments = json.loads(tool_call.function.arguments)
                result = self._execute_tool(tool_call.function.name, arguments)

                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                tool_results.append({
                    "tool": tool_call.function.name,
                    "result": json.loads(result)
                })

            # Get final response after tool execution
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.conversation_history

            final_response = client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            final_content = final_response.choices[0].message.content
            self.conversation_history.append({
                "role": "assistant",
                "content": final_content
            })

            return AgentResponse(
                content=final_content,
                tool_calls=tool_results,
                raw_response=final_response
            )
        else:
            # No tool calls, just return the response
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content
            })

            return AgentResponse(
                content=assistant_message.content,
                raw_response=response
            )

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []


class ScreeningAgent(ValueInvestmentAgent):
    """Specialized agent for stock screening tasks."""

    def __init__(self):
        super().__init__()
        self.system_prompt = """You are a stock screening specialist AI.
Your job is to help investors find stocks that match their criteria.

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
"""


class AnomalyAgent(ValueInvestmentAgent):
    """Specialized agent for anomaly detection and forensic analysis."""

    def __init__(self):
        super().__init__()
        self.system_prompt = """You are a forensic accounting and anomaly detection specialist AI.
Your job is to analyze companies for financial red flags and anomalies.

When analyzing:
1. Explain what each quality metric means (M-Score, Z-Score, F-Score)
2. Distinguish between legitimate business reasons and concerning patterns
3. Provide specific investigation recommendations
4. Rate the overall risk level and explain why

Key thresholds:
- M-Score > -1.78: Higher probability of earnings manipulation
- Z-Score < 1.8: Financial distress zone
- F-Score < 3: Weak financial position
- Sloan Ratio > 10%: Earnings quality concerns
"""


class ResearchAgent(ValueInvestmentAgent):
    """Specialized agent for investment research and thesis generation."""

    def __init__(self):
        super().__init__()
        self.system_prompt = """You are an investment research analyst AI.
Your job is to provide in-depth analysis and investment thesis for stocks.

When analyzing:
1. Evaluate both bull and bear cases
2. Identify key metrics to monitor
3. Assess margin of safety
4. Provide clear buy/hold/avoid recommendations with reasoning

Focus on value investing principles:
- Look for undervalued opportunities
- Emphasize earnings power and sustainability
- Consider balance sheet strength
- Factor in competitive position
"""


# ============================================================================
# Convenience Functions
# ============================================================================

def get_agent(agent_type: str = "general") -> ValueInvestmentAgent:
    """
    Get an agent instance.

    Args:
        agent_type: Type of agent ('general', 'screening', 'anomaly', 'research')

    Returns:
        Agent instance
    """
    agents = {
        "general": ValueInvestmentAgent,
        "screening": ScreeningAgent,
        "anomaly": AnomalyAgent,
        "research": ResearchAgent
    }

    agent_class = agents.get(agent_type, ValueInvestmentAgent)
    return agent_class()


def quick_screen(market: str, **criteria) -> str:
    """
    Quick screening with AI analysis.

    Args:
        market: 'US' or 'SG'
        **criteria: Screening criteria

    Returns:
        AI analysis of screening results
    """
    agent = ScreeningAgent()
    criteria_str = ", ".join(f"{k}={v}" for k, v in criteria.items())
    prompt = f"Screen {market} stocks with these criteria: {criteria_str}. Analyze the results and recommend top picks."
    response = agent.chat(prompt)
    return response.content


def quick_analyze(symbol: str) -> str:
    """
    Quick anomaly analysis with AI interpretation.

    Args:
        symbol: Stock ticker symbol

    Returns:
        AI analysis of anomalies
    """
    agent = AnomalyAgent()
    prompt = f"Analyze {symbol} for financial anomalies and red flags. Provide a detailed risk assessment."
    response = agent.chat(prompt)
    return response.content


# ============================================================================
# Main / Demo
# ============================================================================

if __name__ == "__main__":
    # Demo the agent system
    print("=" * 60)
    print("Value Investment AI Agent Demo")
    print("=" * 60)

    agent = ValueInvestmentAgent()

    # Test screening
    print("\n[Test 1: Stock Screening]")
    response = agent.chat("Screen US stocks with ROE > 15% and gross margin > 30%")
    print(f"Response: {response.content[:500]}...")

    # Test anomaly detection
    print("\n[Test 2: Anomaly Detection]")
    agent.reset_conversation()
    response = agent.chat("Analyze DDI for any financial anomalies")
    print(f"Response: {response.content[:500]}...")

    print("\n" + "=" * 60)
    print("Demo complete!")
