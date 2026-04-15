import streamlit as st

st.markdown("# Literature Review Demo")
st.write("Click the button below to initialize the demo settings and proceed to the Literature Review.")

if st.button("Start Demo", type="primary"):
    st.session_state.selected_datasets = ['Fire Weather Index (FWI) projections', 'Seasonal Temperature Maximum projections', 'Precipitation projections']
    st.session_state.questions = [
        "How do changes in wildfire risk perception and public policy influence property values in areas susceptible to wildfires?", 
        "How have historical wildfire events and subsequent property value changes in similar metropolitan areas influenced policy decisions regarding infrastructure mitigation strategies?"
    ]
    st.session_state.custom_goals = [
        "Analyze the historical trends of FWI in Denver, CO and identify areas with significant increases or decreases in fire risk.",
        "Investigate scientific literature to understand how changes in wildfire risk perception and public policy have affected property values in areas similar to Denver.",
        "Explore the relationship between changes in wildfire risk (as reflected by FWI projections) and current property value assessments and insurance premiums in different areas of Denver."
    ]
    st.session_state.responses= {
        "Location": "Denver, CO",
        "Profession": "Risk Manager",
        "Concern": "Property values",
        "Timeline": "30 - 50 years",
        "Scope": "changes might affect property values in different areas based on their proximity to fire risk zones and existing infrastructure mitigation strategies."
    }
    st.session_state.questions_done = True
    st.session_state.profile_done = True
    
    st.switch_page("experience/literature_review.py")
