import streamlit as st
import pandas as pd

from functions.resume import render_sentiment_chart


def render():
    st.title("Gráficos de análise de sentimentos")
    st.markdown("---")

    insight = st.session_state.history
    df = pd.DataFrame(insight)

    column_labels = {
        "url":        "URL",
        "summary":    "Resumo",
        "sentimento": "Sentimento",
        "feedback":   "Feedback",
    }

    if "feedback" in df.columns:

        st.subheader("Distribuição de Feedback")

        feedback_counts = (
            df["feedback"]
            .value_counts()
            .rename_axis("Tipo")
            .reset_index(name="Quantidade")
            .set_index("Tipo")
        )

        st.bar_chart(feedback_counts)
      
    if "sentimento" in df.columns:

        st.subheader("🧠 Distribuição de Sentimentos")

        sentiment_counts = (
            df["sentimento"]
            .value_counts()
            .rename_axis("Sentimento")
            .reset_index(name="Quantidade")
            .set_index("Sentimento")
        )

        st.bar_chart(sentiment_counts)

        st.markdown("---")
      
        st.markdown("---")
        st.subheader("Gráfico de Distribuição")
      
        counts = df["sentimento"].value_counts().to_dict()
        dist_global = {
            "positive": counts.get("Positivo", 0),
            "neutral": counts.get("Neutro", 0),
            "negative": counts.get("Negativo", 0)
        }

        if any(dist_global.values()):
            render_sentiment_chart(dist_global)
 
    st.subheader("🔎 Visualizar análise individual")

    options = {
        f"Análise {i + 1} — {row['url'][:60]}...": i
        for i, row in df.iterrows()
    }

    selected_label = st.selectbox("Selecionar análise", list(options.keys()))
    selected_idx   = options[selected_label]

    row = df.loc[selected_idx]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sentimento", row.get("sentimento", "N/A"))

    with col2:
        st.metric("Feedback",   row.get("feedback",   "N/A"))

    st.subheader("Resumo")
    st.write(row["summary"])
