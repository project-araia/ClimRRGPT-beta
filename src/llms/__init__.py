import ollama
from abc import ABC, abstractmethod
import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.registry import AgentRegistry

class ChatCompletion(ABC):
    def __init__(self, **args):
        pass

    @abstractmethod
    def get_response(self, messages, options, content = True, stream = False):
        pass


class OpenAI(ChatCompletion):
    def __init__(self, model=None, **args):
        super().__init__(**args)
        import os
        from openai import OpenAI as OpenAIClient
        from inference_auth_token import get_access_token
        from dotenv import load_dotenv

        load_dotenv()
        
        self.model = model or os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")
        # self.client = OpenAIClient(
        #     api_key=get_access_token(),
        #     base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        # )
        self.client = OpenAIClient(
            api_key="jnavarro",
            base_url="https://apps.inside.anl.gov/argoapi/v1",
        )

    def get_response(self, messages, options=None, content=True, stream=False, stream_handler=None):
        if options is None:
            options = {}
            
        # Flatten messages to ensure content is a string for vLLM
        flat_messages = []
        for msg in messages:
            content_val = msg.get("content", "")
            if isinstance(content_val, list):
                text_pieces = []
                for chunk in content_val:
                    if isinstance(chunk, str):
                        text_pieces.append(chunk)
                    elif isinstance(chunk, dict) and chunk.get("type") == "text":
                        text_pieces.append(chunk.get("text", ""))
                flat_messages.append({**msg, "content": "\n".join(text_pieces)})
            else:
                flat_messages.append(msg)
        messages = flat_messages

        kwargs = {}
        if "temperature" in options: kwargs["temperature"] = options["temperature"]
        if "max_tokens" in options: kwargs["max_tokens"] = options["max_tokens"]
        if "top_p" in options: kwargs["top_p"] = options["top_p"]

        if stream:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs
            )
            response = ''
            if stream_handler:
                def adapter_stream():
                    for chunk in response_stream:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            yield {"message": {"content": chunk.choices[0].delta.content}}
                response = stream_handler(adapter_stream())
            else:
                message_placeholder = st.empty()
                for chunk in response_stream:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        content_piece = chunk.choices[0].delta.content
                        response += content_piece
                        if "<think>" in response:
                            message_placeholder.markdown("LLM is thinking...")
                            if "</think>" in response:
                                response_clean = response.split("</think>")[1]
                                message_placeholder.markdown(response_clean)
                        else:
                            message_placeholder.markdown(response)
            return response
        else:
            response_obj = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                **kwargs
            )
            if content:
                return response_obj.choices[0].message.content
            else:
                return {
                    "message": {
                        "content": response_obj.choices[0].message.content,
                        "role": response_obj.choices[0].message.role,
                    }
                }

class OpenSourceModels(ChatCompletion):
    def __init__(self, model, **args):
        super().__init__(**args)
        self.model = model

    def get_response(self, messages, options, content = True, stream = False, stream_handler = None):
        if stream:
            stream = ollama.chat(model=self.model, messages=messages, stream=stream, options=options)
            response = ''
            if stream_handler:
                response = stream_handler(stream)
            else:
                message_placeholder = st.empty()
                for chunk in stream:
                    response += chunk['message']['content']
                    if "<think>" in response:
                        # add small text display of thinking process
                        message_placeholder.markdown("LLM is thinking...")
                        if "</think>" in response:
                            # response comes after </think>
                            response = response.split("</think>")[1]
                            message_placeholder.markdown(response)
                    else:
                        message_placeholder.markdown(response)
            return response
        else:
            if content:
                return ollama.chat(model=self.model, messages=messages, options=options)['message']['content']
            else:
                return ollama.chat(model=self.model, messages=messages, options=options)

class OpenSourceVisionModels(ChatCompletion):
    # TODO
    def __init__(self, model, **args):
        super().__init__(**args)
        self.model = model

    def get_response(self, messages, options, content = True, stream = False, stream_handler = None):
        if stream:
            stream = ollama.chat(model=self.model, messages=messages, stream=stream, options=options)
            response = ''
            if stream_handler:
                response = stream_handler(stream)
            else:
                message_placeholder = st.empty()
                for chunk in stream:
                    response += chunk['message']['content']
                    message_placeholder.markdown(response)
            return response
        else:
            if content:
                return ollama.chat(model=self.model, messages=messages, options=options)['message']['content']
            else:
                return ollama.chat(model=self.model, messages=messages, options=options)


class OpenSourceCodingModels(ChatCompletion):
    def __init__(self, model, **args):
        super().__init__(**args)
        self.model = model

    def get_response(self, messages, options, content = True, stream = False, stream_handler = None):
        if stream:
            stream = ollama.chat(model=self.model, messages=messages, stream = stream, options=options)
            response = ''
            if stream_handler:
                response = stream_handler(stream)
            else:
                return ollama.chat(model=self.model, messages=messages, options=options)


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

    # Fallback to the overarching Ollama model
    messages = context + history + [{"role": "user", "content": prompt}]
    
    # We display the thinking/streaming via the original get_response_fn
    with st.chat_message("assistant"):
        response = get_response_fn(
            messages=messages,
            stream=True,
            options={"top_p": 0.9, "max_tokens": 2048, "temperature": 0.7}
        )
        
    return response, registry.DEFAULT