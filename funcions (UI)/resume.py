import matplotlib.pyplot as plt
import streamlit as st

def render_sentiment_chart(distribuicao: dict):
    """
    Cria a figura do gráfico. Usa st.pyplot aqui e retorna ele pronto.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    
    ax.pie(
        distribuicao.values(),
        labels=distribuicao.keys(),
        autopct='%1.1f%%',
        colors=colors,
        startangle=140
    )
    ax.set_title("Distribuição de Sentimentos")
     
    return st.pyplot(fig) 
