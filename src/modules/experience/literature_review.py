import os
import streamlit as st

from src.utils import load_config
from src.llms import OpenSourceModels, route_and_respond, get_default_llm
from src.literature.search import literature_search
from src.agents.registry import AgentRegistry
from src.agents.demo_agents import register_demo_agents
from src.agents.history import ChatHistory


def initialize_session_state(config_path: str):
    """Initializes common session state variables and configurations."""
    if "config" not in st.session_state:
        st.session_state.config = load_config(config_path)

    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    # Build context if demographic info is present but context isn't built yet
    if (
        "context" not in st.session_state 
        and "responses" in st.session_state 
        and "selected_datasets" in st.session_state
    ):
        responses = st.session_state.responses
        datasets = st.session_state.selected_datasets
        st.session_state.context = [
            {
                "role": "user",
                "content": f"Here is my profile:\n\nProfession: {responses['Profession']}\n\nConcern: {responses['Concern']}\n\nLocation: {responses['Location']}\n\nTimeline: {responses['Timeline']}\n\nScope: {responses['Scope']}",
            },
            {
                "role": "user",
                "content": f"Let's review these datasets:\n\n{[datasets[i] for i in range(len(datasets))]}",
            },
        ]
        if "analysis" in st.session_state:
            st.session_state.context.append({"role": "assistant", "content": st.session_state.analysis})
        if "data_analysis_summary" in st.session_state:
            st.session_state.context.append({"role": "assistant", "content": st.session_state.data_analysis_summary})

    # Initialize Multi-Agent Components
    if "agent_registry" not in st.session_state:
        reg = AgentRegistry()
        register_demo_agents(reg)
        st.session_state.agent_registry = reg

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatHistory(session_id="literature_chat")
        # Load past sessions into history UI
        st.session_state.qa_messages = st.session_state.chat_history.load_messages()


def render_header(config: dict):
    """Renders the top welcome and instructions components."""
    st.write(config["welcome_message"])
    with st.expander("Instructions"):
        st.write(config["instruction_message"])


def handle_search_and_generation(state: st.session_state, get_response_fn):
    """Handles logic for literature search and the initial summary generation."""
    if "literature_review_summary" in state:
        return # Skip if already done

    state.retrieved_literature = []
    state.references = []

    # Clean up empty slots
    if state.questions[1] == "":
        state.questions.pop(1)

    st.markdown("#### Question (s):")

    with st.spinner("Loading. Please do not click any buttons or refresh the page."):
        for i, question in enumerate(state.questions):
            st.write(f"**{question}**")
            retrieved, refs = literature_search(question)
            with st.expander("Show Literature Search Results"):
                st.write(retrieved)
            state.retrieved_literature.append(retrieved)
            state.references += refs

    state.context.append({
        "role": "system",
        "content": f"Here are the reference papers: {state.retrieved_literature}.",
    })

    if st.button("Generate Summary", use_container_width=True):
        messages = [
            {
                "role": "system",
                "content": state.config["literature_review_instructions"][0]["content"].format(
                    retrieve_literature=state.retrieved_literature
                ),
            }
        ]
        state.context.append({
            "role": "user",
            "content": f"Here are my questions:\\n\\n{[state.questions[i] for i in range(len(state.questions))]}",
        })

        messages = state.context + messages

        summary = get_response_fn(
            messages=messages,
            stream=True,
            options={
                "top_p": 0.9,
                "max_tokens": 2048,
                "temperature": 0.7,
                "stop": ["Works Cited\n", "References\n", "Bibliography\n"],
            },
        )
        
        # Clean up citations
        for stop_word in ["Works Cited", "References", "Bibliography"]:
            if stop_word in summary:
                summary = summary.split(stop_word)[0]

        # Process unique references
        refs_unique = sorted(list(dict.fromkeys(state.references)))
        summary += "\n\n### References:\n\n"
        for ref in refs_unique:
            summary += f"{ref}\n\n"

        state.literature_review_summary = summary
        state.context.append({"role": "assistant", "content": summary})
        st.rerun()


def render_summary_display(state: st.session_state):
    """Renders the generated summary and its related search results if present."""
    if "literature_review_summary" not in state:
        return

    st.markdown("#### Question (s):")
    for i, question in enumerate(state.questions):
        st.write(f"**{question}**")
        with st.expander("Show Literature Search Results"):
            st.write(state.retrieved_literature[i])
    st.write(state.literature_review_summary)
    st.markdown("---")


def render_chat_interface(state: st.session_state, get_response_fn):
    """Displays chat history and input widget for multi-agent interaction."""
    st.markdown("#### Chat with Literature Review Assistant")

    # Clear History Button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            state.chat_history.clear()
            state.qa_messages = []
            st.rerun()

    if "literature_review_summary" not in state:
        st.info("💡 The literature review is still in progress above. You can already start chatting — relevant papers will be retrieved live for each message.")

    # Show History
    for message in state.qa_messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            agent = message.get("agent_name", "default")
            if agent != "default":
                st.markdown(f"*(🤖 via {agent})*\n\n{content}")
            else:
                st.markdown(content)

    # Chat Input Handler
    if prompt := st.chat_input("How can I help you?"):
        st.chat_message("user").markdown(prompt)
        
        state.qa_messages.append({"role": "user", "content": prompt, "agent_name": "default"})
        state.chat_history.save_message("user", prompt, "default")
        
        clean_history = [{"role": m["role"], "content": m["content"]} for m in state.qa_messages[:-1]]

        response, agent_name = route_and_respond(
            prompt=prompt,
            history=clean_history,
            registry=state.agent_registry,
            context=state.context,
            get_response_fn=get_response_fn,
        )

        state.qa_messages.append({"role": "assistant", "content": response, "agent_name": agent_name})
        state.chat_history.save_message("assistant", response, agent_name)
        st.rerun()


def main():
    """Main application entry point."""
    initialize_session_state("src/modules/experience/literature_review.yml")
    state = st.session_state
    
    get_response = get_default_llm(state)

    # Pre-flight Check
    if ("questions_done" not in state) or (not state.questions_done):
        if st.button("Identify Questions"):
            st.switch_page("experience/question_identification.py")
        return

    # Main UI Flow
    render_header(state.config)
    handle_search_and_generation(state, get_response)
    render_summary_display(state)
    render_chat_interface(state, get_response)


if __name__ == "__main__":
    main()
