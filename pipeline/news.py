from providers.web_scrapper import (
    coleta, 
    preparacao, 
    analise_local
)

def analyze_news(url: str):
    
    df_bruto = coleta([url])
    
    if df_bruto.empty:
        return None

    df_final = preparacao(df_bruto)

    if not df_final.empty:
        resultado_analise = analise_local(df_final)
        return {
            "article": df_final.iloc[0]['texto_bruto'],  
            "summary": resultado_analise['summary'],      
            "sentiment": {
                "label": resultado_analise['overall_sentiment'], 
                "score": resultado_analise['polarity_val'],      
                "distribution": resultado_analise['distribution'], 
                "emoji": "😊" if resultado_analise['overall_sentiment'] == "Positivo" else "😐"
            }
        }
    
    return None