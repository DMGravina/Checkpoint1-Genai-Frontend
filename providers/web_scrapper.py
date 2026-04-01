import requests
import pandas as pd
import re
import os
import matplotlib
import matplotlib.pyplot as plt
import nltk
import statistics

from bs4 import BeautifulSoup
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

matplotlib.use('Agg')  

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("Ambiente configurado com sucesso (Modo Offline).")

df_final_global = None   
analise_global  = None   

def coleta(urls: list) -> pd.DataFrame:
    dataset_bruto = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for url in urls:
        try:
            print(f"🔍 [RPA] Coletando: {url}")
            res = requests.get(url, headers=headers, timeout=15)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')
            fragments = [tag.text.strip() for tag in soup.find_all(['p', 'h1', 'h2'])]
            content   = " ".join([f for f in fragments if len(f) > 30])

            if len(content) > 100:
                dataset_bruto.append({"url": url, "texto_bruto": content})

        except Exception as e:
            print(f"[Erro] Falha em {url}: {e}")

    return pd.DataFrame(dataset_bruto)

def preparacao(df: pd.DataFrame) -> pd.DataFrame:
    
    if df.empty:
        return df

    print("🧹 [Processamento] Normalizando dados...")

    def limpar_texto(texto):
        texto = texto.lower()
        texto = re.sub(r'[^a-zá-ú0-9\s\.]', '', texto) 
        return re.sub(r'\s+', ' ', texto).strip()

    df['texto_limpo'] = df['texto_bruto'].apply(limpar_texto)
    df = df.drop_duplicates(subset=['texto_limpo'])
    df = df[df['texto_limpo'].str.len() > 150]

    df.to_csv("dataset_estruturado.csv", index=False)
    return df




def analise_local(df: pd.DataFrame) -> dict:
    print(" [NLP Local] Iniciando processamento estatístico melhorado...")

    # 1. MELHORIA NO SENTIMENTO:
    # Para português sem usar modelos pesados, uma técnica é traduzir ou usar 
    # thresholds mais sensíveis. Aqui, vamos garantir que estamos analisando 
    # o texto original (com contexto) se disponível.
    
    coluna_texto = 'texto' if 'texto' in df.columns else 'texto_limpo'
    
    sentiment_scores = []
    for txt in df[coluna_texto]:
        # Tenta traduzir para o léxico do TextBlob ou use uma lógica de sensibilidade
        blob = TextBlob(txt)
        # Se o TextBlob não for traduzido, ele terá dificuldade com PT-BR.
        # Uma alternativa rápida é o uso de bibliotecas como 'vaderSentiment-pt'
        sentiment_scores.append(blob.sentiment.polarity)

    avg_polarity = statistics.mean(sentiment_scores) if sentiment_scores else 0

    # Ajuste de Threshold (Mais sensível para detectar negatividade)
    if avg_polarity > 0.1:
        overall = "Positivo"
    elif avg_polarity < -0.1:
        overall = "Negativo"
    else:
        overall = "Neutro"

    # 2. MELHORIA NO RESUMO:
    # Usamos o texto original para manter a pontuação e estrutura.
    # Se 'texto_limpo' não tem pontuação, o sent_tokenize não funciona.
    texto_para_resumo = " ".join(df[coluna_texto].astype(str).tolist())
    sentencas = sent_tokenize(texto_para_resumo)
    
    # Seleciona frases que tenham um tamanho mínimo para evitar "lixo"
    resumo_filtrado = [s.strip() for s in sentencas if len(s) > 30]
    resumo = " ".join(resumo_filtrado[:3]) + "..." if resumo_filtrado else "Resumo indisponível."

    return {
        "overall_sentiment": overall,
        "polarity_val":      round(avg_polarity, 4),
        "summary":           resumo,
        "distribution": {
            "positive": len([s for s in sentiment_scores if s > 0.1]) / len(sentiment_scores) * 100,
            "neutral":  len([s for s in sentiment_scores if -0.1 <= s <= 0.1]) / len(sentiment_scores) * 100,
            "negative": len([s for s in sentiment_scores if s < -0.1]) / len(sentiment_scores) * 100,
        }
    }

 
URLS_PADRAO = [
    "https://www.cnnbrasil.com.br/tecnologia/",
    "https://g1.globo.com/tecnologia/"
]

def run_pipeline(urls: list = None) -> tuple:
    global df_final_global, analise_global

    urls = urls or URLS_PADRAO

    print("\n" + "=" * 60)
    print("ETAPA 1 — Coleta via Scraping")
    print("=" * 60)
    df_bruto = coleta(urls)

    print("\n" + "=" * 60)
    print("ETAPA 2 — Limpeza e Preparação")
    print("=" * 60)
    df_final_global = preparacao(df_bruto)

    if not df_final_global.empty:
        print("\n" + "=" * 60)
        print("ETAPA 3 — Análise NLP Local")
        print("=" * 60)
        analise_global = analise_local(df_final_global)
    else:
        print("❌ Nenhum dado coletado para análise.")
        analise_global = {}

    return df_final_global, analise_global


def get_df_final(urls: list = None) -> pd.DataFrame:
    global df_final_global
    if df_final_global is None or df_final_global.empty:
        print("Dataset não encontrado em memória — iniciando pipeline...")
        run_pipeline(urls=urls)
    return df_final_global


def get_analise(urls: list = None) -> dict:
    global analise_global
    if analise_global is None:
        run_pipeline(urls=urls)
    return analise_global


if __name__ == "__main__":
    print("Executando pipeline completo...")
    df, resultado = run_pipeline()
    print(f"\nPipeline finalizado. {len(df)} documentos processados.")