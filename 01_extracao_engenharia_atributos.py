"""
Etapa 1: Ingestão de Documentos em PDF, Segmentação Heurística
e Engenharia de Atributos (Geração da Variável 'context')
"""

import os
import re
import fitz  # PyMuPDF
import pandas as pd
import spacy

# Carregamento do modelo de língua portuguesa para normalização
nlp = spacy.load("pt_core_news_lg")


def normalizar_texto(texto: str) -> str:
    """Aplica lematização e limpeza inicial com o spaCy."""
    doc = nlp(texto)
    lemas = [token.lemma_ for token in doc if not token.is_space]
    return " ".join(lemas)


def extrair_propostas_pdf(caminho_pdf: str, metadata: dict) -> pd.DataFrame:
    """Extrai texto bruto de ficheiros PDF eleitorais e segmenta em propostas atómicas."""
    doc = fitz.open(caminho_pdf)
    texto_completo = " ".join([pagina.get_text() for pagina in doc])

    # Segmentação heurística por marcadores enumerativos e alíneas
    padrao_propostas = r"(?:\n\d+\.|\n•|\n-)\s*(.*?)(?=(?:\n\d+\.|\n•|\n-|\Z))"
    itens = re.findall(padrao_propostas, texto_completo, re.DOTALL)

    registros = []

    # Rotina principal: listas estruturadas
    if len(itens) > 0:
        for idx, item in enumerate(itens):
            item_limpo = re.sub(r"\s+", " ", item).strip()

            # Filtro sintático para expurgar cabeçalhos, rodapés e ruídos
            if len(item_limpo) > 25:
                partes = item_limpo.split(",", 1)
                titulo = partes[0].strip()
                descricao = partes[1].strip() if len(partes) > 1 else partes[0].strip()

                # Engenharia de atributos: fusão com token separador [SEP]
                context_feature = f"{titulo} [SEP] {descricao}"

                registros.append(
                    {
                        "id_proposta": f"{metadata['uf']}_{idx+1}",
                        "titulo_proposta": titulo,
                        "descricao_proposta": descricao,
                        "context": context_feature,
                        **metadata,
                    }
                )
    else:
        # Rotina de contingência (fallback narrativo): quebra por parágrafos
        paragrafos = texto_completo.split("\n\n")
        for idx, p in enumerate(paragrafos):
            p_limpo = re.sub(r"\s+", " ", p).strip()
            if len(p_limpo) > 25:
                context_feature = f"{p_limpo} [SEP] {p_limpo}"
                registros.append(
                    {
                        "id_proposta": f"{metadata['uf']}_{idx+1}",
                        "titulo_proposta": p_limpo[:60],
                        "descricao_proposta": p_limpo,
                        "context": context_feature,
                        **metadata,
                    }
                )

    return pd.DataFrame(registros)


if __name__ == "__main__":
    print("Módulo de extração e engenharia de atributos carregado com sucesso.")
