"""
Pre-built demo agents and the function to register them.

Add new agents here as the system grows. Each agent is independent —
just provide a name, trigger keywords/patterns, and a handler function.
"""

from __future__ import annotations

from src.agents.registry import Agent, AgentRegistry


# ── Foobar agent (proof-of-concept demo) ─────────────────────────────────────

def _foobar_handler(message: str, history: list[dict]) -> str:
    return (
        "👋 You mentioned **foobar**! I'm the foobar agent.\n\n"
        "In a real deployment I would perform a specialized foobar task. "
        "For now, here's a placeholder response so you can see routing in action.\n\n"
        f"_(Your original message: \"{message}\")_"
    )


foobar_agent = Agent(
    name="foobar_agent",
    trigger_patterns=["foobar"],
    handler=_foobar_handler,
    description="Demo agent — triggers on 'foobar' keyword.",
)


# ── Literature agent ──────────────────────────────────────────────────────────

def _make_literature_handler():
    """Lazily import literature_search to avoid loading the model at module import."""
    def handler(message: str, history: list[dict]) -> str:
        from src.literature.search import literature_search
        retrieved, references = literature_search(message)
        if not retrieved.strip():
            return "I searched the literature database but couldn't find closely relevant papers for that query."
        return (
            "📚 I searched the literature database and found these relevant papers:\n\n"
            + retrieved
            + "\n\n**References:**\n\n"
            + "".join(references)
        )
    return handler


literature_agent = Agent(
    name="literature_agent",
    trigger_patterns=[
        r"\bpaper\b",
        r"\bpapers\b",
        r"\bresearch\b",
        r"\bliterature\b",
        r"\bcitation\b",
        r"\bstudy\b",
        r"\bstudies\b",
        r"\bjournal\b",
        r"\bpublication\b",
        r"\bscientific\b",
    ],
    handler=_make_literature_handler(),
    description="Routes literature/research queries to the resilience paper vector index.",
)


# ── Deep reasoning agent ──────────────────────────────────────────────────────

def _make_deep_agent_handler():
    """Lazily import the deep agent handler."""
    def handler(message: str, history: list[dict]) -> str:
        from src.agents.deep_agent import deep_agent_handler
        return deep_agent_handler(message, history)
    return handler

deep_reasoning_agent = Agent(
    name="deep_reasoning_agent",
    trigger_patterns=[
        r"\bplan\b",
        r"\banalyze\b",
        r"\banalysis\b",
        r"\bdeep\b",
        r"\breasoning\b",
        r"\bcomplex\b",
        r"\bexplain\b",
    ],
    handler=_make_deep_agent_handler(),
    description="Complex multi-step reasoning agent using a 120b model.",
)

# ── Registration helper ───────────────────────────────────────────────────────

def register_demo_agents(registry: AgentRegistry) -> None:
    """Register all demo agents into the given registry."""
    # Order matters: first match wins.
    registry.register(foobar_agent)
    registry.register(literature_agent)
    registry.register(deep_reasoning_agent)
