import streamlit as st 
import pipeline.news
from functions.resume import render_sentiment_chart


def render():
    st.title("🔍 Análise de Notícias com IA")
    st.markdown("Insira a URL de uma notícia e a IA irá extrair, processar e analisar o conteúdo.")
    st.markdown("---")

    st.subheader("1. Informe a URL")

    url = st.text_input(
        "URL da notícia",
        placeholder="https://g1.globo.com/...",
        key="url_input"    
    )

    st.button(
        "Executar análise",
        on_click=run_analysis,
        type="primary"
    )

    if not st.session_state.summary:
        st.info("⬆ Insira uma URL acima e clique em **Executar análise** para começar.")
        return

    st.markdown("---")
    st.subheader("2. Resultados")

    
    tab_sentiment, tab_summary, tab_raw = st.tabs([
        "Sentimento",
        "Resumo",
        "Texto extraído",
    ])

    with tab_sentiment:

        sentiment = st.session_state.sentiment

        if sentiment:
            st.subheader("Sentimento detectado na notícia")

            col_emoji, col_label, col_score = st.columns([1, 2, 2])

            with col_emoji:
                st.markdown(
                    f"<h1 style='text-align:center'>{sentiment['emoji']}</h1>",
                    unsafe_allow_html=True
                )

            with col_label:
                st.metric(
                    label="Classificação",
                    value=sentiment["label"]
                )

            with col_score:
                st.metric(
                    label="Confiança do modelo",
                    value=f"{abs(sentiment['score']) * 100:.0f}%"
                )

            st.progress(abs(sentiment["score"]))

            st.caption(
                "A análise de sentimento indica o tom predominante da notícia "
                "com base no conteúdo textual extraído."
            )
 
        else:
            st.info("Sentimento não disponível para esta análise.")

    with tab_summary:

        st.subheader("Resumo gerado pelo modelo")

        placeholder = st.empty()
        displayed_text = ""

        for word in st.session_state.summary.split():
            displayed_text += word + " "
            placeholder.write(displayed_text)

        st.markdown("---")

        st.subheader("Esse resumo foi útil?")

        col_pos, col_neg = st.columns(2)

        with col_pos:
            if st.button("👍 Útil"):
                _save_feedback("positivo")
                st.success("Obrigado pelo feedback positivo!")

        with col_neg:
            if st.button("👎 Ruim"):
                _save_feedback("negativo")
                st.error("Obrigado por nos avisar! Vamos melhorar.")

    with tab_raw:

        st.subheader("Texto extraído da notícia")
        st.caption("Conteúdo bruto capturado pelo scraper antes do processamento.")

        st.text_area(
            "Conteúdo",
            value=st.session_state.article_text,
            height=350,
            disabled=True    

def _save_feedback(feedback_type: str):
    st.session_state.history.append({
        "url":       st.session_state.current_url,
        "summary":   st.session_state.summary,
        "sentimento": st.session_state.sentiment["label"] if st.session_state.sentiment else "N/A",
        "feedback":  feedback_type,
    })

 
def run_analysis():
    url = st.session_state.get("url_input", "").strip()
    if not url:
        st.warning("Insira uma URL.")
        return

    # Chama o pipeline (que agora está em /pipelines)
    result = analyze_news(url=url)

    if result:
        st.session_state.article_text = result["article"]
        st.session_state.summary      = result["summary"]
        st.session_state.sentiment    = result["sentiment"]
        st.session_state.current_url  = url
    else:
        st.error("Não foi possível analisar esta URL.")