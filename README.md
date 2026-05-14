# Zap News ⚡ - Projeto Integrado de Front-end e PLN

Este repositório contém uma aplicação web interativa desenvolvida em Python com **Streamlit**. O objetivo central é integrar uma interface de usuário modular com um pipeline de Processamento de Linguagem Natural (PLN) para análise automatizada de conteúdo jornalístico.

O sistema permite a extração de notícias, processamento via modelos de linguagem para análise de sentimento, e a geração de resumos textuais, separando estritamente as responsabilidades de Front-end e Back-end (NLP/Providers).

## 🏗️ Arquitetura e Separação de Responsabilidades

O projeto foi construído seguindo uma arquitetura modularizada, focada em componentização e roteamento, atendendo aos requisitos da disciplina de Front-end. A estrutura isola a lógica de interface, o pipeline de dados de PLN e os serviços externos (providers).

### Estrutura de Diretórios

```text
📦 Checkpoint2-NLP-Frontend
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
