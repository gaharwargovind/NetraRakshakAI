"""
Streamlit Web Application Shell for Rural PHC Screening and Tele-Ophthalmology Triage.
Prototype interface scaffolding (Phase 1).
"""

import streamlit as st

st.set_page_config(
    page_title="AI DR Screening (SIH26038)",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("Navigation")
st.sidebar.info(
    "Rural Health DR Screening Assistant\n"
    "Problem Statement: SIH26038\n"
    "Status: Prototype Scaffolding (Phase 1)"
)

st.title("Explainable AI for Diabetic Retinopathy Screening")
st.markdown(
    """
    ### Primary Care Tele-Screening Portal
    This system assists Primary Health Centre (PHC) health workers in identifying 
    referable Diabetic Retinopathy from color fundus photographs.
    
    * **Screening Workflow**: Patient Intake → IQA → AI Inference → Report Generation
    * **Review Queue**: Dedicated tele-ophthalmology verification for uncertain cases
    * **System Analytics**: Workflow throughput and quality monitoring
    """
)
st.warning(
    "Clinical Notice: This decision-support prototype is currently under active development. "
    "ML models and inference pipelines will be activated in subsequent phases."
)