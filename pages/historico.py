import streamlit as st
import pandas as pd

from functions.resume import render_sentiment_chart


def render():
    st.title("Histórico de Análises")
    st.markdown("Consulte todas as notícias analisadas nesta sessão.")
    st.markdown("---")

    history = st.session_state.history

    if len(history) == 0:
        st.info("Nenhuma análise registrada ainda. Vá para **Analisar notícia** para começar.")
        return
      
    df = pd.DataFrame(history)

    column_labels = {
        "url":        "URL",
        "summary":    "Resumo",
        "sentimento": "Sentimento",
        "feedback":   "Feedback",
    }
    df_display = df.rename(columns=column_labels)

    st.subheader(f"📊 {len(history)} análise(s) registrada(s)")
    st.dataframe(df_display, use_container_width=True)
