"""
One-shot Claude completion helper using claude-agent-sdk.

All LLM calls in this project route through here so they use the local
Claude Code CLI subscription auth (no ANTHROPIC_API_KEY needed).

Implementation notes:
- Each call runs on its own thread + fresh event loop so the SDK's
  subprocess is fully torn down before the next call begins.
- Uses ClaudeSDKClient (the same pattern that works for the AI Chatbot
  page) rather than query() — more robust for sequential calls.
- Has a small retry-with-backoff to recover from transient
  "Command failed with exit code 1" subprocess errors.

Prerequisites:
    1. Node.js installed
    2. Claude Code CLI installed
    3. `claude /login` run once to set up subscription auth
"""
import asyncio
import threading
import queue as queue_mod
import time
from typing import Optional

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)


DEFAULT_MODEL = "claude-sonnet-4-6"


async def claude_complete_async(
    user: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = 5,
) -> str:
    """Single-turn Claude completion via ClaudeSDKClient."""
    options = ClaudeAgentOptions(
        system_prompt=system,
        model=model,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        # Empty allowlist forces a pure text completion. Without this the
        # CLI exposes its full tool palette (WebSearch, Bash, etc.) and the
        # model burns its single turn on tool calls instead of answering.
        allowed_tools=[],
    )

    chunks = []
    async with ClaudeSDKClient(options=options) as client:
        await client.query(user)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(message, ResultMessage):
                break

    return "".join(chunks).strip()


def _run_in_thread(coro_factory) -> str:
    """Run a coroutine in a dedicated thread with its own event loop."""
    result_q: queue_mod.Queue = queue_mod.Queue(maxsize=1)

    def _worker():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            text = loop.run_until_complete(coro_factory())
            result_q.put(("ok", text))
        except BaseException as e:
            result_q.put(("err", e))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()

    status, payload = result_q.get_nowait()
    if status == "err":
        raise payload
    return payload


def claude_complete(
    user: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = 5,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> str:
    """
    Sync wrapper. Runs the async call in a dedicated thread with its own
    event loop. Auto-retries up to `max_retries` times on subprocess errors.
    """
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return _run_in_thread(
                lambda: claude_complete_async(
                    user=user, system=system, model=model, max_turns=max_turns
                )
            )
        except BaseException as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_delay * (attempt + 1))  # 2s, 4s
                continue
            raise last_error


# ============================================================================
# Web-enabled completion (for tasks that need real-time web data, e.g. 10-year
# financials lookup). Uses more turns + WebSearch tool. Slower and burns more
# subscription quota — use sparingly.
# ============================================================================

async def claude_complete_with_websearch_async(
    user: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = 30,
) -> str:
    """Multi-turn Claude call with WebSearch + WebFetch tools enabled."""
    options = ClaudeAgentOptions(
        system_prompt=system,
        model=model,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        allowed_tools=["WebSearch", "WebFetch"],
    )

    chunks = []
    async with ClaudeSDKClient(options=options) as client:
        await client.query(user)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(message, ResultMessage):
                break

    return "".join(chunks).strip()


def claude_complete_with_websearch(
    user: str,
    system: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = 30,
    max_retries: int = 1,
    retry_delay: float = 3.0,
) -> str:
    """Sync wrapper around claude_complete_with_websearch_async.

    Note: max_turns defaults to 30 because WebSearch tasks need many turns
    (search → read → extract → search next year → ...). 10 was too few.
    """
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return _run_in_thread(
                lambda: claude_complete_with_websearch_async(
                    user=user, system=system, model=model, max_turns=max_turns
                )
            )
        except BaseException as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise last_error


def is_available() -> bool:
    """Check if claude-agent-sdk import succeeded (basic availability check)."""
    return True  # Import at module load — if we got here, it's available
