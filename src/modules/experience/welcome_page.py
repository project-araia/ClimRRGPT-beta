import streamlit as st

st.markdown("Get ready to ignite your understanding of hazard risks with ClimRRGPT!\n\nClimRRGPT guides you through understanding and addressing hazard risks, especially for assessing future risks in specific locations. ", unsafe_allow_html=True)

# read welcome message from `welcome.md`
with open("src/modules/welcome.md", "r") as f:
    instruction_message = f.read()

with st.expander("Here's your journey:"):
    st.markdown(instruction_message, unsafe_allow_html=True)

# Display the start button
col1, col2, col3 = st.columns([1,2,1])

with col2:
    if st.button("Start Your ClimRRGPT Journey", use_container_width=True):
        st.switch_page("experience/profile.py")
