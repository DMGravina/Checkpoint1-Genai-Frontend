import streamlit as st 

def render_sidebar():
    st.sidebar.title("⚡News")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Páginas")
    
    if st.sidebar.button("Análise da Notícia", use_container_width=True):
        st.session_state.page = "ana"
    
    if st.sidebar.button("Histórico", use_container_width=True):
        st.session_state.page = "his"
        
    if st.sidebar.button("Insights", use_container_width=True):
        st.session_state.page = "ins"
         
    return st.session_state.page