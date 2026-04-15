import streamlit as st


def model_picker_page():
    st.title("⚙️ Model Configuration")
    st.markdown(
        "Customize your AI experience by selecting between local processing or external high-performance models."
    )

    # provider selection
    provider_options = ["Local", "ALCF", "OpenAI"]
    current_provider = st.session_state.get("llm_provider", "Local")

    # Ensure index is valid for radio bootstrap
    try:
        provider_idx = provider_options.index(current_provider)
    except ValueError:
        provider_idx = 0

    provider = st.radio(
        "Select Model Provider",
        provider_options,
        index=provider_idx,
        help="Local uses Ollama. ALCF uses models hosted on Argonne computers and requires authentication. "
        + "OpenAI uses standard cloud models.",
    )

    st.session_state["llm_provider"] = provider

    # Provider-specific settings
    if provider == "OpenAI":
        # API Key management
        col1, col2 = st.columns([3, 1])
        with col1:
            custom_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get("custom_api_key", ""),
                type="password",
                placeholder="sk-...",
                help="Pasted keys are stored in session memory only and prioritized over .env files.",
            )
            st.session_state["custom_api_key"] = custom_key
        with col2:
            st.write("##")  # alignment
            if st.button("🗑️ Clear Key", use_container_width=True):
                st.session_state["custom_api_key"] = ""
                st.rerun()

    # Model name logic per provider
    if provider == "Local":
        known_models = ["gemma4:e4b", "gemma3:4b"]
    elif provider == "ALCF":
        known_models = ["openai/gpt-oss-120b", "google/gemma-3-27b-it"]
    else:  # OpenAI
        known_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]

    current_model = st.session_state.get("llm_model_name", known_models[0])

    # Handle the "Other" logic
    options = known_models + ["Other"]

    # Try to find the current model in the options
    if current_model in known_models:
        default_idx = known_models.index(current_model)
    else:
        default_idx = len(options) - 1  # Point to "Other"

    selection = st.selectbox(f"Select {provider} Model", options, index=default_idx)

    if selection == "Other":
        other_model = st.text_input(
            "Enter Custom Model Name",
            value=current_model if current_model not in known_models else "",
            placeholder="e.g. gemma:3b",
        )
        if other_model:
            st.session_state["llm_model_name"] = other_model
    else:
        st.session_state["llm_model_name"] = selection

    st.divider()

    # Show active configuration status
    status_text = f"**Provider:** {provider} | **Model:** `{st.session_state.get('llm_model_name')}`"
    if provider == "OpenAI" and st.session_state.get("custom_api_key"):
        status_text += " | **Key:** `SET 🔒`"

    st.info(status_text)

    if st.button("Save & Continue", type="primary"):
        st.success("Configuration saved! You can now proceed to any Analysis module.")


if __name__ == "__main__":
    model_picker_page()
