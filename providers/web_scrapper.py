import requests
import pandas as pd
import re
import os
import matplotlib
import matplotlib.pyplot as plt
import nltk
import statistics

from leiabr import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
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
    print(" [NLP Local] Iniciando processamento estatístico em PT-BR...")

    # 1. CORREÇÃO DA COLUNA: 
    # Usamos o 'texto_bruto' que possui vírgulas, pontos e letras maiúsculas.
    # Isso é vital tanto para o resumo quanto para o contexto do sentimento.
    coluna_texto = 'texto_bruto' if 'texto_bruto' in df.columns else 'texto_limpo'
    
    # 2. CORREÇÃO DO IDIOMA:
    # Instanciamos o analisador nativo em Português
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []
    
    for txt in df[coluna_texto]:
        # O LeIA retorna um dicionário. O 'compound' é a pontuação final de -1 a 1.
        score = analyzer.polarity_scores(str(txt))
        sentiment_scores.append(score['compound'])

    avg_polarity = statistics.mean(sentiment_scores) if sentiment_scores else 0

    # O threshold padrão da literatura para o VADER/LeIA é 0.05
    if avg_polarity >= 0.05:
        overall = "Positivo"
    elif avg_polarity <= -0.05:
        overall = "Negativo"
    else:
        overall = "Neutro"

    # 3. CORREÇÃO DO RESUMO:
    # Agora operando sobre o texto_bruto, o sent_tokenize saberá exatamente
    # onde uma frase começa e termina por causa dos pontos finais originais.
    texto_para_resumo = " ".join(df[coluna_texto].astype(str).tolist())
    sentencas = sent_tokenize(texto_para_resumo)
    
    resumo_filtrado = [s.strip() for s in sentencas if len(s) > 30]
    resumo = " ".join(resumo_filtrado[:3]) + "..." if resumo_filtrado else "Resumo indisponível."

    return {
        "overall_sentiment": overall,
        "polarity_val":      round(avg_polarity, 4),
        "summary":           resumo,
        "distribution": {
            "positive": len([s for s in sentiment_scores if s >= 0.05]) / len(sentiment_scores) * 100 if sentiment_scores else 0,
            "neutral":  len([s for s in sentiment_scores if -0.05 < s < 0.05]) / len(sentiment_scores) * 100 if sentiment_scores else 0,
            "negative": len([s for s in sentiment_scores if s <= -0.05]) / len(sentiment_scores) * 100 if sentiment_scores else 0,
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