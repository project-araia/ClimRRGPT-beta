"""
Agent registry — the central routing broker.

Usage:
    from src.agents.registry import AgentRegistry, Agent

    registry = AgentRegistry()
    registry.register(Agent(
        name="foobar_agent",
        trigger_patterns=["foobar"],
        handler=my_handler_fn,
    ))
    agent_name = registry.route("tell me about foobar")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Agent:
    """A named agent with keyword triggers and a callable handler."""

    name: str
    # Plain strings (case-insensitive substring match) or regex patterns.
    trigger_patterns: list[str]
    # handler(user_message, chat_history) -> response_string
    handler: Callable[[str, list[dict]], str]
    description: str = ""

    def should_handle(self, message: str) -> bool:
        """Return True if any trigger pattern matches the message."""
        lower = message.lower()
        for pattern in self.trigger_patterns:
            try:
                if re.search(pattern, lower):
                    return True
            except re.error:
                # Treat malformed regex as a plain substring
                if pattern.lower() in lower:
                    return True
        return False


class AgentRegistry:
    """Registry that maps agent names to Agent objects and routes messages."""

    DEFAULT = "default"

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        # Registration order matters — first match wins.
        self._order: list[str] = []

    def register(self, agent: Agent) -> None:
        """Register an agent. Re-registration overwrites the previous entry."""
        if agent.name not in self._agents:
            self._order.append(agent.name)
        self._agents[agent.name] = agent

    def route(self, message: str) -> str:
        """Return the name of the first matching agent, or 'default'."""
        for name in self._order:
            if self._agents[name].should_handle(message):
                return name
        return self.DEFAULT

    def get(self, name: str) -> Agent | None:
        return self._agents.get(name)

    @property
    def agents(self) -> dict[str, Agent]:
        return self._agents
