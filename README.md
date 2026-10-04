# Análise de Políticas Públicas e Igualdade de Gênero nos Programas de Governo da Região Norte (Eleições 2026)

Repositório com os dados, códigos e rotinas analíticas desenvolvidos para o Trabalho de Conclusão de Curso do **MBA em Inteligência Artificial e Big Data** do **Instituto de Ciências Matemáticas e de Computação da Universidade de São Paulo (ICMC-USP)**.

- **Autora:** Ana Carolina Moura Fernandes de Almeida
- **Orientador:** Prof. Dr. Fábio Manoel França Lobato

---

## 📁 Estrutura do Repositório

- `01_extracao_engenharia_atributos.py`: Ingestão dos programas de governo em PDF, segmentação heurística por regex/contingência e construção da variável contextual `context` (`[SEP]`).
- `02_modelagem_bertimbau_hitl.py`: Adaptação prévia de domínio eleitoral via MLM, ajuste fino supervisionado (*Human-in-the-Loop*) e cálculo das métricas de desempenho comparativo.
- `propostas_governo_norte_2026.csv`: Base tratada contendo os 1.172 registros de propostas eleitorais catalogadas.

---

## 🛠️ Tecnologias Empregadas
- Python 3
- PyMuPDF (`fitz`) e Expressões Regulares (`re`)
- spaCy (`pt_core_news_lg`)
- PyTorch e Hugging Face Transformers (`neuralmind/bert-base-portuguese-cased`)
- Scikit-learn e Pandas
