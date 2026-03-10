"""
Advanced reasoning agent powered by DeepAgents and a 120b model via the ALCF inference API.
This agent can handle multi-step planning and tool-based research.
"""

import os
from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from inference_auth_token import get_access_token
from dotenv import load_dotenv

class FlatteningChatOpenAI(ChatOpenAI):
    """
    vLLM strictly requires string content for messages. deepagents (and LangChain) 
    sometimes passes content as a list of dicts (for multimodal/complex prompts).
    This subclass intercepts the messages and flattens them back into strings.
    """
    def _flatten_messages(self, messages):
        flat_messages = []
        for msg in messages:
            if isinstance(msg.content, list):
                text_blocks = []
                for block in msg.content:
                    if isinstance(block, str):
                        text_blocks.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        text_blocks.append(block.get("text", ""))
                
                # Copy the message but with flat string content
                if hasattr(msg, "model_copy"):
                    new_msg = msg.model_copy(update={"content": "\n".join(text_blocks)})
                else:
                    new_msg = msg.copy(update={"content": "\n".join(text_blocks)})
                flat_messages.append(new_msg)
            else:
                flat_messages.append(msg)
        return flat_messages

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._generate(self._flatten_messages(messages), stop=stop, run_manager=run_manager, **kwargs)
        
    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._stream(self._flatten_messages(messages), stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await super()._agenerate(self._flatten_messages(messages), stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        return await super()._astream(self._flatten_messages(messages), stop=stop, run_manager=run_manager, **kwargs)

def get_deep_agent_executor():
    """
    Initialize and return the DeepAgent executor.
    Uses the ALCF inference API and a large (120b) model.
    """
    load_dotenv()
    
    model_name = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
    
    # Initialize the robust model wrapper
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
