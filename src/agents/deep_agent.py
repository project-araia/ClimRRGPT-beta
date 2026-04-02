"""
Advanced reasoning agent powered by DeepAgents and a 120b model via the ALCF inference API.
This agent can handle multi-step planning and tool-based research.
"""

import os
from deepagents import create_deep_agent
from inference_auth_token import get_access_token
from dotenv import load_dotenv

# Standardize with the shared model abstraction
from src.llms import FlatteningChatOpenAI

def get_deep_agent_executor():
    """
    Initialize and return the DeepAgent executor.
    Uses the ALCF inference API and a large (120b) model.
    """
    load_dotenv()
    
    model_name = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
    
    # Initialize the robust model wrapper relocated to src.llms
    model = FlatteningChatOpenAI(
        model=model_name,
        api_key=get_access_token(),
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    )
    
    # Create the agent with its default batteries-included tools (planning, filesystem, etc.)
    return create_deep_agent(model=model)

def deep_agent_handler(message: str, history: list[dict]) -> str:
    """
    Agent handler that delegates work to the DeepAgents orchestrated graph.
    """
    executor = get_deep_agent_executor()
    
    # Map raw chat history to LangChain messages if possible, or just pass prompt
    # For now, let's just pass the user prompt to the invoke flow.
    # Note: invoke is safer than stream with some local vLLM backends.
    try:
        result = executor.invoke({"messages": [{"role": "user", "content": message}]})
        final_messages = result.get("messages", [])
        if final_messages:
            return final_messages[-1].content
        return "The specialized reasoning agent failed to produce a response."
    except Exception as e:
        return f"⚠️ DeepAgent error: {str(e)}"
