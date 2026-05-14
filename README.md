# Zap News⚡- Leitor de notícias
>Projeto Integrado de Front-end e PLN

## Informações
Este repositório contém uma aplicação web interativa desenvolvida em Python com **Streamlit**. O objetivo central é integrar uma interface de usuário modular com um pipeline de Processamento de Linguagem Natural (PLN) para análise automatizada de conteúdo jornalístico.

O sistema permite a extração de notícias, processamento via modelos de linguagem para análise de sentimento, e a geração de resumos textuais, separando estritamente as responsabilidades de Front-end e Back-end (NLP/Providers).

## Arquitetura e Separação de Responsabilidades

O projeto foi construído seguindo uma arquitetura modularizada, focada em componentização e roteamento, atendendo aos requisitos da disciplina de Front-end. A estrutura isola a lógica de interface, o pipeline de dados de PLN e os serviços externos (providers).

### Estrutura de Diretórios

```text
📦 Leitor-de-noticias
 ┣ 📂 fases/              # Telas roteadas da aplicação (Páginas)
 ┃ ┣ 📜 analise.py        # Interface principal de input e output da notícia
 ┃ ┣ 📜 historico.py      # Tela do diferencial: Histórico de interações salvas no Session State
 ┃ ┗ 📜 insights.py       # Tela para visualizações extras ou comparações
 ┣ 📂 functions/          # Componentes de UI modulares e reutilizáveis
 ┃ ┣ 📜 SideBar.py        # Renderização da barra lateral e controle de navegação
 ┃ ┗ 📜 resume.py         # Componente de formatação/exibição do resumo gerado
 ┣ 📂 pipeline/           # Lógica de negócio e integração de PLN
 ┃ ┗ 📜 news.py           # Pipeline que recebe o texto, chama os modelos e consolida saídas
 ┣ 📂 providers/          # Serviços externos e conectores de dados
 ┃ ┗ 📜 web_scrapper.py   # Script de extração da notícia (herdado/adaptado do CP1)
 ┣ 📂 state/              # Gerenciamento de estado da aplicação (Streamlit Session State)
 ┃ ┗ 📜 session.py        # Inicialização e controle de variáveis globais (ex: histórico)
 ┣ 📜 app.py              # Entry-point da aplicação, gerencia configuração da página e rotas
 ┣ 📜 pacotes.txt         # Lista de dependências (requirements)
 ┗ 📜 README.md           # Documentação do projeto
```
## Funcionalidades Implementadas

### 1. Processamento de Linguagem Natural (PLN)
* **Obtenção do Conteúdo:** Extração de texto de notícias via web scraping ou input direto (adaptado do modelo base do CP1).
* **Análise de Sentimento:** Processamento do texto bruto da notícia por modelo de NLP para classificação de sentimento (polaridade).
* **Sumarização de Texto:** Geração de um resumo sintético do conteúdo original utilizando técnicas de modelagem de linguagem.

### 2. Interface e Front-end (Streamlit)
* **Roteamento Multi-page:** Navegação estruturada no `app.py` orquestrando a renderização das telas de análise, histórico e insights (`fases.analise`, `fases.historico` e `fases.insights`).
* **Gerenciamento de Estado (State Management):** Uso de variáveis de sessão do Streamlit para armazenar e exibir o histórico das interações e notícias consultadas sem perda de contexto entre as páginas.
* **Modularidade:** Componentização da interface isolando elementos visuais reutilizáveis no diretório `functions/` (como o componente da barra lateral).

---

## Como Executar o Projeto Localmente

1. Clone este repositório para o seu ambiente local:

   ```bash
   git clone <URL_DO_SEU_REPOSITORIO>
   cd Checkpoint2-NLP-Frontend
   ```
2. Crie e ative um ambiente virtual para isolar as dependências do projeto:

   ```bash
   python -m venv venv
   
   # Ativação no Windows:
   venv\Scripts\activate
   
   # Ativação no Linux/macOS:
   source venv/bin/activate
   ```

3. Instale as bibliotecas e dependências necessárias:

   ```bash
   pip install -r pacotes.txt
   ```

4. Suba o servidor da aplicação Streamlit:

   ```bash
   streamlit run app.py
   ```
