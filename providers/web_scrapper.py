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

# from IPython.display import display, Markdown
# IPython.display é exclusivo do ambiente Jupyter/Colab.
# No VS Code usamos print() para texto e retornamos figuras do matplotlib
# para que o Streamlit possa renderizá-las com st.pyplot().

# Matplotlib em modo não-interativo: evita que plt.show() abra janela
# separada ao rodar via Streamlit (o Streamlit renderiza a figura via st.pyplot).
# No terminal standalone, show() continua funcionando normalmente.
matplotlib.use('Agg')  # sem isso, plt.show() trava em alguns ambientes sem display

# Download de recursos essenciais do NLTK (execução local)
# quiet=True suprime a saída verbosa — os arquivos ficam em ~/nltk_data
nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("✅ Ambiente configurado com sucesso (Modo Offline).")


# =============================================================================
# Variáveis de módulo — preenchidas após run_pipeline()
# Ficam expostas para importação pelo news_pipeline.py
# =============================================================================
df_final_global = None   # DataFrame limpo e estruturado
analise_global  = None   # Dicionário com resultado da análise NLP


# =============================================================================
# ETAPA 1 — Coleta via Scraping (RPA com requests + BeautifulSoup)
# =============================================================================

def coleta(urls: list) -> pd.DataFrame:
    """
    Faz scraping das URLs informadas e extrai o conteúdo textual relevante.

    Args:
        urls (list): Lista de URLs a coletar.

    Returns:
        pd.DataFrame: Colunas ["url", "texto_bruto"] com o conteúdo coletado.
    """
    dataset_bruto = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for url in urls:
        try:
            print(f"🔍 [RPA] Coletando: {url}")
            res = requests.get(url, headers=headers, timeout=15)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'html.parser')

            # Extrai parágrafos e títulos significativos
            fragments = [tag.text.strip() for tag in soup.find_all(['p', 'h1', 'h2'])]
            content   = " ".join([f for f in fragments if len(f) > 30])

            if len(content) > 100:
                dataset_bruto.append({"url": url, "texto_bruto": content})

        except Exception as e:
            print(f"❌ [Erro] Falha em {url}: {e}")

    return pd.DataFrame(dataset_bruto)


# =============================================================================
# ETAPA 2 — Preparação e Limpeza do Texto
# =============================================================================

def preparacao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza o texto bruto, remove duplicatas e exporta CSV estruturado.

    Args:
        df (pd.DataFrame): DataFrame com coluna "texto_bruto".

    Returns:
        pd.DataFrame: DataFrame com coluna "texto_limpo" adicionada.
    """
    if df.empty:
        return df

    print("🧹 [Processamento] Normalizando dados...")

    def limpar_texto(texto):
        texto = texto.lower()
        texto = re.sub(r'[^a-zá-ú0-9\s\.]', '', texto)  # Mantém letras, números e pontos
        return re.sub(r'\s+', ' ', texto).strip()

    df['texto_limpo'] = df['texto_bruto'].apply(limpar_texto)
    df = df.drop_duplicates(subset=['texto_limpo'])
    df = df[df['texto_limpo'].str.len() > 150]

    df.to_csv("dataset_estruturado.csv", index=False)
    return df


# =============================================================================
# ETAPA 3 — Análise NLP Local (Sentimento + TF-IDF + Sumarização)
# =============================================================================

def analise_local(df: pd.DataFrame) -> dict:
    """
    Executa análise NLP completa sobre o DataFrame limpo:
      1. Sentimento via TextBlob (polaridade léxica)
      2. Extração de temas via TF-IDF
      3. Sumarização extrativa simples

    Args:
        df (pd.DataFrame): DataFrame com coluna "texto_limpo".

    Returns:
        dict: Resultado completo da análise com as chaves:
              overall_sentiment, polarity_val, themes, summary, distribution
    """
    print("⚙️ [NLP Local] Iniciando processamento estatístico...")

    texto_completo = " ".join(df['texto_limpo'].tolist())

    # 1. Análise de Sentimento (Polaridade)
    # Nota: TextBlob em PT-BR funciona melhor com tradução ou léxicos simples.
    # Aqui usamos polaridade média dos documentos.
    sentiment_scores = [TextBlob(txt).sentiment.polarity for txt in df['texto_limpo']]
    avg_polarity     = sum(sentiment_scores) / len(sentiment_scores)

    overall = (
        "Positivo" if avg_polarity >  0.05 else
        "Negativo" if avg_polarity < -0.05 else
        "Neutro"
    )

    # 2. Extração de Temas (TF-IDF)
    vectorizer  = TfidfVectorizer(max_features=10, stop_words=stopwords.words('portuguese'))
    tfidf_matrix = vectorizer.fit_transform(df['texto_limpo'])
    temas       = vectorizer.get_feature_names_out()

    # 3. Sumarização Extrativa Simples
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

 

# =============================================================================
# PIPELINE PRINCIPAL — run_pipeline()
# =============================================================================

# URLs padrão — podem ser sobrescritas ao chamar run_pipeline(urls=[...])
URLS_PADRAO = [
    "https://www.cnnbrasil.com.br/tecnologia/",
    "https://g1.globo.com/tecnologia/"
]

def run_pipeline(urls: list = None) -> tuple:
    """
    Executa o pipeline completo de coleta e análise NLP.

    Args:
        urls (list): Lista de URLs para coletar. Se None, usa URLS_PADRAO.

    Returns:
        tuple: (df_final, resultado_analise)
               df_final        → pd.DataFrame com texto limpo estruturado
               resultado_analise → dict com sentimento, temas e sumarização
    """
    global df_final_global, analise_global

    urls = urls or URLS_PADRAO

    # ── Etapa 1: Coleta ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📡 ETAPA 1 — Coleta via Scraping")
    print("=" * 60)
    df_bruto = coleta(urls)

    # ── Etapa 2: Preparação ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("🧹 ETAPA 2 — Limpeza e Preparação")
    print("=" * 60)
    df_final_global = preparacao(df_bruto)

    # ── Etapa 3: Análise NLP ──────────────────────────────────────────────────
    if not df_final_global.empty:
        print("\n" + "=" * 60)
        print("🤖 ETAPA 3 — Análise NLP Local")
        print("=" * 60)
        analise_global = analise_local(df_final_global)
    else:
        print("❌ Nenhum dado coletado para análise.")
        analise_global = {}

    return df_final_global, analise_global


# =============================================================================
# Funções auxiliares — interface para o news_pipeline.py
# =============================================================================

def get_df_final(urls: list = None) -> pd.DataFrame:
    global df_final_global
    if df_final_global is None or df_final_global.empty:
        print("📡 Dataset não encontrado em memória — iniciando pipeline...")
        run_pipeline(urls=urls)
    return df_final_global


def get_analise(urls: list = None) -> dict:
    global analise_global
    if analise_global is None:
        run_pipeline(urls=urls)
    return analise_global


# só executa quando chamado diretamente pelo terminal.
if __name__ == "__main__":
    print("🚀 Executando pipeline completo...")
    df, resultado = run_pipeline()
    print(f"\n🏁 Pipeline finalizado. {len(df)} documentos processados.")