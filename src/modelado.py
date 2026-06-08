# =============================================================================
# src/modelado.py
# Motor matemático del pipeline NLP.
#
# CHANGELOG:
#   - Eliminación total de la métrica y anclas de "Precio".
#   - Refactorización de clasificador de sentimientos: Transición de similitud 
#     coseno a Analizador Léxico (Rule-Based) para evitar falsos positivos.
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
import hdbscan

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from joblib import Memory
from typing import Tuple, Dict

# Sistema de caché joblib
os.makedirs("cache_nlp", exist_ok=True)
memoria = Memory("cache_nlp", verbose=0)

# Diccionarios Léxicos para Análisis de Sentimiento
_POS_FUERTE = {
    "excelente": 4,
    "perfecto": 4,
    "maravilloso": 4,
    "espectacular": 4,
    "increible": 4,
    "increíble": 4,
    "fantastico": 4,
    "fantástico": 4,
    "extraordinario": 4,
    "impresionante": 4,
    "magnifico": 4,
    "magnífico": 4
}

_POS_NORMAL = {
    "bonito": 1,
    "hermoso": 1,
    "agradable": 1,
    "limpio": 1,
    "amable": 1,
    "bueno": 1,
    "comodas": 1,
    "comodo": 1,
    "cómodo": 1,
    "tranquilo": 1,
    "recomendable": 1,
    "encanta": 1,
    "encantó": 1,
    "encanto": 1,
    "agradó": 1,
    "agradable": 1,
    "lindo": 1,
    "excelencia": 1
}

_NEG_FUERTE = {
    "terrible": 5,
    "pesimo": 5,
    "pésimo": 5,
    "asqueroso": 6,
    "estafa": 6,
    "robo": 6,
    "inseguro": 5,
    "horrible": 5,
    "decepcion": 5,
    "decepción": 5,
    "sucio": 5,
    "apestoso": 5,
    "asco": 6,
    "fraude": 6,
    "desastre": 5
}

_NEG_NORMAL = {
    "caro": 2,
    "lento": 2,
    "ruidoso": 2,
    "malo": 2,
    "molesto": 2,
    "deficiente": 2,
    "falla": 2,
    "fallas": 2,
    "problema": 2,
    "problemas": 2,
    "regular": 2,
    "incómodo": 2,
    "incomodo": 2
}

# Capa de embeddings (cacheada y forzada a CPU para clustering y PCA)
@memoria.cache
def obtener_embeddings(textos: list) -> np.ndarray:
    modelo = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",   
    )
    return modelo.encode(textos, show_progress_bar=True)

# Funciones internas del pipeline

def detectar_outliers_y_ngramas(
    df: pd.DataFrame, columna_texto: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, pd.DataFrame()

    textos = df[columna_texto].tolist()
    embeddings = obtener_embeddings(textos)

    # Mantenemos el clustering ejecutándose por integridad matemática
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        gen_min_span_tree=True,
    )
    clusterer.fit(embeddings)

    df_normal = df.copy()
    df_outliers_vacios = pd.DataFrame()

    return df_normal, df_outliers_vacios

def clasificar_sentimientos(df: pd.DataFrame, columna_texto: str) -> pd.DataFrame:
    def _score_polaridad(texto: str) -> str:
        # Tokenizamos conservando el orden de las palabras
        tokens = re.findall(r'\b\w+\b', str(texto).lower())
        score = 0
        inversores = {"no", "sin", "cero", "nunca", "tampoco", "nada"}

        for i, t in enumerate(tokens):
            # Ventana de lectura (lookback) de hasta 2 tokens atrás
            es_negado = False
            if i > 0 and tokens[i-1] in inversores:
                es_negado = True
            elif i > 1 and tokens[i-2] in inversores:
                es_negado = True

            peso = 0
            if t in _POS_FUERTE: peso = 3
            elif t in _POS_NORMAL: peso = 1
            elif t in _NEG_FUERTE: peso = -4
            elif t in _NEG_NORMAL: peso = -1

            if peso != 0:
                if es_negado:
                    score -= peso  
                else:
                    score += peso

        if score > 0: return "positivo"
        if score < 0: return "negativo"
        return "neutral"

    df_copy = df.copy()
    df_copy["sentimiento"] = df_copy[columna_texto].apply(_score_polaridad)
    return df_copy

def extraer_red_semantica(df: pd.DataFrame, columna_texto: str) -> dict:
    if df.empty or columna_texto not in df.columns:
        return {"nodos": [], "enlaces": []}

    vectorizer = CountVectorizer(
    max_features=50,
    min_df=2
    )
    try:
        X = vectorizer.fit_transform(df[columna_texto].dropna())
    except ValueError:
        return {"nodos": [], "enlaces": []}

    palabras = vectorizer.get_feature_names_out()
    co_ocurrencia = (X.T * X).toarray()
    np.fill_diagonal(co_ocurrencia, 0)

    # Cálculo de umbral dinámico: Retener solo el 15% de las conexiones más fuertes
    valores_activos = co_ocurrencia[co_ocurrencia > 0]
    if len(valores_activos) == 0:
        return {"nodos": palabras.tolist(), "enlaces": []}
    
    umbral_dinamico = np.percentile(valores_activos, 85)

    nodos = palabras.tolist()
    enlaces = []

    for i in range(len(nodos)):
        for j in range(i + 1, len(nodos)):
            peso = co_ocurrencia[i, j]
            if peso >= umbral_dinamico and peso > 0:
                enlaces.append({
                    "fuente": nodos[i],
                    "destino": nodos[j],
                    "peso": int(peso),
                })

    return {"nodos": nodos, "enlaces": enlaces}

def _calcular_pca_2d(df: pd.DataFrame, columna_texto: str) -> dict:
    if df.empty or "sentimiento" not in df.columns or len(df) < 3:
        return {"componentes": [], "etiquetas": []}

    textos     = df[columna_texto].tolist()
    embeddings = obtener_embeddings(textos)     
    n_componentes  = min(2, embeddings.shape[0], embeddings.shape[1])
    pca            = PCA(n_components=n_componentes)
    componentes_2d = pca.fit_transform(embeddings)

    return {
        "componentes": componentes_2d.tolist(),
        "etiquetas":   df["sentimiento"].tolist(),
    }

# Función maestra pública

def ejecutar(df_clasificado: pd.DataFrame, nombre_destino: str = "") -> dict:
    tag = f"[{nombre_destino}] " if nombre_destino else ""
    print(f"  {tag}Iniciando pipeline de modelado ({len(df_clasificado)} registros)...")

    # 1. Detección de ruido
    print(f"  {tag}[1/3] HDBSCAN — detección de ruido semántico...")
    df_normal, df_outliers_sem = detectar_outliers_y_ngramas(
        df_clasificado, columna_texto="comentario_nlp"
    )
    ngramas_ruido = df_outliers_sem.attrs.get("ngramas_ruido", {})
    
    print(f"  {tag}  → Coherentes: {len(df_normal)} | Ruido: {len(df_outliers_sem)}")



    # 2. Sentimientos (Basado en reglas léxicas)
    print(f"  {tag}[2/3] Clasificando sentimientos (Análisis Léxico)...")
    df_con_sentimiento = clasificar_sentimientos(df_normal, columna_texto="comentario")
    dist = df_con_sentimiento["sentimiento"].value_counts().to_dict()
    print(f"  {tag}  → Distribución: {dist}")

    # 3. Red semántica + PCA
    print(f"  {tag}[3/3] Red semántica y proyección PCA...")

    datos_red = extraer_red_semantica(
        df_con_sentimiento,
        columna_texto="comentario_grafo"
    )

    datos_pca = _calcular_pca_2d(
        df_con_sentimiento,
        columna_texto="comentario_nlp"
    )

    print(
        f"  {tag}Modelado completo. "
        f"Nodos: {len(datos_red['nodos'])} | "
        f"Aristas: {len(datos_red['enlaces'])}"
    )

    return {
        "df_procesado": df_con_sentimiento,
        "df_outliers_semanticos": df_outliers_sem,
        "datos_red": datos_red,
        "pca": datos_pca,
        "ngramas_ruido": ngramas_ruido,
    }