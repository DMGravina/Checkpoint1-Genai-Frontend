import streamlit as st

def init_session():
 
    if "page" not in st.session_state:
        st.session_state.page = "ana"
    
    if "history" not in st.session_state:
        st.session_state.history = []

    if "summary" not in st.session_state:
        st.session_state.summary = None

    if "sentiment" not in st.session_state:
        st.session_state.sentiment = None

    if "article_text" not in st.session_state:
        st.session_state.article_text = ""

    if "current_url" not in st.session_state:
        st.session_state.current_url = ""

    if "df_final" not in st.session_state:
        st.session_state.df_final = None
        
    if "analise" not in st.session_state:
        st.session_state.analise = None        