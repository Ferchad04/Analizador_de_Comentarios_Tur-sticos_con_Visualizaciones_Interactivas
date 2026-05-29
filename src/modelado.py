# src/modelado.py
import pandas as pd
import numpy as np
import hdbscan
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from joblib import Memory
from typing import Tuple, Dict

# 1. Configuración del sistema de caché local (Joblib)
os.makedirs("cache_nlp", exist_ok=True)
memoria = Memory("cache_nlp", verbose=0)

@memoria.cache
def obtener_embeddings(textos: list) -> np.ndarray:
    """
    Calcula los embeddings y los guarda en disco. 
    Si los mismos textos se procesan de nuevo, carga la matriz cacheada instantáneamente.
    """
    # Se utiliza un modelo multilingüe eficiente para garantizar soporte offline en es, en, fr
    modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    return modelo.encode(textos, show_progress_bar=True)

def detectar_outliers_y_ngramas(df: pd.DataFrame, columna_texto: str) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # Detecta anomalías semánticas aplicando HDBSCAN directamente sobre los embeddings.

    if df.empty:
        return df, pd.DataFrame()

    # Extraer matriz vectorial de los textos utilizando el sistema de caché para evitar cálculos repetidos.
    textos = df[columna_texto].tolist()
    embeddings = obtener_embeddings(textos)
    
    # 2. Implementación de HDBSCAN en lugar de Isolation Forest
    # min_cluster_size ajusta la agresividad del filtrado. 
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, gen_min_span_tree=True)
    etiquetas = clusterer.fit_predict(embeddings)
    
    df_copy = df.copy()
    # HDBSCAN etiqueta automáticamente el ruido/outliers con -1
    df_copy['outlier_label'] = etiquetas
    
    df_normal = df_copy[df_copy['outlier_label'] != -1].drop(columns=['outlier_label'])
    df_outliers = df_copy[df_copy['outlier_label'] == -1].copy()
    
    if not df_outliers.empty:
        # Extracción de unigramas y bigramas sobre el ruido para entender por qué son anomalías
        vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=10)
        X = vectorizer.fit_transform(df_outliers[columna_texto])
        frecuencias = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
        df_outliers.attrs['ngramas_ruido'] = frecuencias
        
    return df_normal, df_outliers

def clasificar_sentimientos(df: pd.DataFrame, columna_texto: str) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['sentimiento'] = np.random.choice(['positivo', 'neutral', 'negativo'], size=len(df))
    return df_copy

def modelar_topicos(df: pd.DataFrame) -> Dict[str, dict]:
    # Mantiene la estructura de retorno esperada por la función de visualización.
    return {"topico_0": {"palabras_clave": ["mock", "data"]}}

def analizar_similitud_precio(df: pd.DataFrame, columna_texto: str) -> pd.DataFrame:
    #Calcula similitud semántica reutilizando los embeddings cacheados.
    if df.empty:
        df['similitud_precio_costo'] = None
        return df

    textos = df[columna_texto].tolist()
    # Esto es O(1) en tiempo si ya pasó por la detección de outliers, gracias al caché
    embeddings_textos = obtener_embeddings(textos)
    
    # Vectorización del concepto estático
    embedding_concepto = obtener_embeddings(["precio valor costo"])
    
    similitudes = cosine_similarity(embeddings_textos, embedding_concepto).flatten()
    
    df_copy = df.copy()
    df_copy['similitud_precio_costo'] = similitudes
    return df_copy