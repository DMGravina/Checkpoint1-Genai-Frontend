import requests
import pandas as pd
import re
import os
import matplotlib
import matplotlib.pyplot as plt
import nltk

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
    print(" [NLP Local] Iniciando processamento estatístico...")

    texto_completo = " ".join(df['texto_limpo'].tolist())
    sentiment_scores = [TextBlob(txt).sentiment.polarity for txt in df['texto_limpo']]
    avg_polarity     = sum(sentiment_scores) / len(sentiment_scores)

    overall = (
        "Positivo" if avg_polarity >  0.05 else
        "Negativo" if avg_polarity < -0.05 else
        "Neutro"
    )

    vectorizer  = TfidfVectorizer(max_features=10, stop_words=stopwords.words('portuguese'))
    tfidf_matrix = vectorizer.fit_transform(df['texto_limpo'])
    temas       = vectorizer.get_feature_names_out()
    sentencas = sent_tokenize(texto_completo)
    resumo    = " ".join(sentencas[:3]) + "..."  # Pega as premissas iniciais dos textos

    return {
        "overall_sentiment": overall,
        "polarity_val":      avg_polarity,
        "themes":            list(temas),
        "summary":           resumo,
        "distribution": {
            "positive": len([s for s in sentiment_scores if s >  0.05]) / len(sentiment_scores) * 100,
            "neutral":  len([s for s in sentiment_scores if -0.05 <= s <= 0.05]) / len(sentiment_scores) * 100,
            "negative": len([s for s in sentiment_scores if s < -0.05]) / len(sentiment_scores) * 100,
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