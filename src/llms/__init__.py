import streamlit as st
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from inference_auth_token import get_access_token

if TYPE_CHECKING:
    from src.agents.registry import AgentRegistry

class ChatCompletion(ABC):
    def __init__(self, **args):
        pass

    @abstractmethod
    def get_response(self, messages: List[Dict], options: Dict, content=True, stream=False, stream_handler=None):
        pass


class FlatteningChatOpenAI(ChatOpenAI):
    """
    vLLM strictly requires string content for messages. deepagents (and LangChain) 
    sometimes passes content as a list of dicts (for multimodal/complex prompts).
    This subclass intercepts the messages and flattens them back into strings.
    """
    def _flatten_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
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


class LangChainModelAdapter(ChatCompletion):
    """
    A unified adapter that uses LangChain's Chat models but exposes the
    get_response() interface used by the Streamlit experiences.
    """
    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        
        # Backward compatibility for 'model' kwarg
        if not model_name and "model" in kwargs:
            model_name = kwargs["model"]
            
        self.provider = provider or "Local"
        self.model_name = model_name or "qwen3.5:latest"
        self.api_key = api_key
        
        if self.provider == "ALCF":
            self.model = FlatteningChatOpenAI(
                model=self.model_name,
                api_key=get_access_token(),
                base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
                streaming=True
            )
        elif self.provider == "OpenAI":
            # Prioritize the UI-pasted key, then environment variables
            effective_key = self.api_key or os.getenv("OPENAI_API_KEY")
            self.model = FlatteningChatOpenAI(
                model=self.model_name,
                api_key=effective_key,
                streaming=True
            )
        else: # Local
            self.model = ChatOllama(
                model=self.model_name,
                streaming=True
            )

    def _convert_messages(self, messages: List[Dict]) -> List[BaseMessage]:
        """Convert list[dict] to list[LangChain message objects]."""
        lc_msgs = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_msgs.append(AIMessage(content=content))
            else:
                lc_msgs.append(HumanMessage(content=content))
        return lc_msgs

    def get_response(self, messages: List[Dict], options: Dict, content=True, stream=False, stream_handler=None):
        lc_messages = self._convert_messages(messages)
        
        # Merge options (like temperature, etc.) into the call
        kwargs = {}
        if options:
            if "temperature" in options: kwargs["temperature"] = options["temperature"]
            if "max_tokens" in options: kwargs["max_tokens"] = options["max_tokens"]
            if "top_p" in options: kwargs["top_p"] = options["top_p"]

        if stream:
            response = ""
            message_placeholder = st.empty()
            
            # Using stream method for better control
            for chunk in self.model.stream(lc_messages, **kwargs):
                content_piece = chunk.content
                response += content_piece
                
                # Support the "thinking" logic if it appears in tags
                if "<think>" in response:
                    message_placeholder.markdown("*(🤖 Thinking...)*")
                    if "</think>" in response:
                        actual_response = response.split("</think>")[-1]
                        message_placeholder.markdown(actual_response)
                else:
                    message_placeholder.markdown(response)
            
            return response
        else:
            res = self.model.invoke(lc_messages, **kwargs)
            if content:
                # To maintain compatibility with your original 'content' flag
                return res.content
            else:
                # Return the full role/content dictionary compatible with legacy expectations
                return {"message": {"role": "assistant", "content": res.content}}

# Maintain aliases for existing callers to reduce breakages while we refactor
OpenSourceModels = LangChainModelAdapter 
OpenSourceVisionModels = LangChainModelAdapter 
OpenSourceCodingModels = LangChainModelAdapter 


def get_default_llm(state: st.session_state) -> Any:
    """
    Utility to resolve the active LLM based on global Model Picker settings.
    Defaults to Local Ollama if no settings are found.
    """
    provider = state.get("llm_provider", "Local")
    api_key = state.get("custom_api_key")
    
    # Fallback to config['model'] if session state is empty (e.g. first load)
    default_model = "qwen3.5:latest"
    if "config" in state and "model" in state.config:
        default_model = state.config["model"]
        
    model_name = state.get("llm_model_name", default_model)
    
    # Instantiate the unified adapter
    adapter = LangChainModelAdapter(provider=provider, model_name=model_name, api_key=api_key)
    return adapter.get_response


def route_and_respond(
    prompt: str,
    history: list[dict],
    registry: "AgentRegistry",
    context: list[dict],
    get_response_fn,
) -> tuple[str, str]:
    """
    Route the user's prompt via the AgentRegistry.
    If an agent handles it, return (agent_response, agent_name).
    Otherwise, fall back to the default LLM and return (llm_response, "default").
    """
    agent_name = registry.route(prompt)

    if agent_name != registry.DEFAULT:
        # Let the specialized agent handle it
        agent = registry.get(agent_name)
        response = agent.handler(prompt, history)
        return response, agent_name

    # Fallback to the overarching model
    messages = context + history + [{"role": "user", "content": prompt}]
    
    with st.chat_message("assistant"):
        response = get_response_fn(
            messages=messages,
            stream=True,
            options={"top_p": 0.9, "max_tokens": 2048, "temperature": 0.7}
        )
        
    return response, registry.DEFAULT