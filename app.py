import streamlit as st
from state.session import init_session          
from ui.sidebar import render_sidebar
from pages import analise as ana
from pages import historico as his
from pages import insights as ins

st.set_page_config(
    page_title="Zap News",
    page_icon="⚡",
    layout="wide"
) 

init_session()

current_page = render_sidebar()

if current_page == "ana":
    ana.render()

elif current_page == "his":
    his.render()

elif current_page == "ins":
    ins.render()