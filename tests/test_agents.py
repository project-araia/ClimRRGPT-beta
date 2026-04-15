from src.agents.registry import Agent, AgentRegistry


def test_agent_substring_matching():
    agent = Agent(
        name="test_agent",
        trigger_patterns=["hello", "world"],
        handler=lambda m, h: "response",
    )
    assert agent.should_handle("Hello there") is True
    assert agent.should_handle("Big World") is True
    assert agent.should_handle("Goodbye") is False


def test_agent_regex_matching():
    agent = Agent(
        name="regex_agent",
        trigger_patterns=[r"user_\d+"],
        handler=lambda m, h: "response",
    )
    assert agent.should_handle("message from user_123") is True
    assert agent.should_handle("message from user_abc") is False


def test_registry_routing():
    registry = AgentRegistry()
    agent1 = Agent(name="agent1", trigger_patterns=["apple"], handler=lambda m, h: "a1")
    agent2 = Agent(
        name="agent2", trigger_patterns=["banana"], handler=lambda m, h: "a2"
    )

    registry.register(agent1)
    registry.register(agent2)

    assert registry.route("I like apples") == "agent1"
    assert registry.route("I like bananas") == "agent2"
    assert registry.route("I like melons") == AgentRegistry.DEFAULT


def test_registry_order():
    registry = AgentRegistry()
    agent1 = Agent(name="agent1", trigger_patterns=["apple"], handler=lambda m, h: "a1")
    agent2 = Agent(name="agent2", trigger_patterns=["apple"], handler=lambda m, h: "a2")

    registry.register(agent1)
    registry.register(agent2)

    # First registered wins
    assert registry.route("Gimme an apple") == "agent1"
