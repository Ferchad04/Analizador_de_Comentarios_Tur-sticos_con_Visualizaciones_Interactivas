import os
# --- BLOQUEO ANTIDEADLOCK PARA MÁQUINAS VIRTUALES WINDOWS ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import hdbscan
import umap.umap_ as umap
from typing import Tuple, Dict
from tqdm import tqdm

# Instancia global para evitar recargas en memoria (Optimización de RAM)
_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer('all-MiniLM-L6-v2')
    return _encoder

def detectar_outliers_y_ngramas(df: pd.DataFrame, columna_texto: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detecta outliers semánticos usando HDBSCAN y extrae n-gramas de ruido.
    Corrige la maldición de la dimensionalidad evaluando sobre UMAP 2D.
    """
    if df.empty: return df, pd.DataFrame()

    encoder = get_encoder()
    
    print("  [FASE 2A] Procesando vectores densos (Embeddings)...")
    embeddings = encoder.encode(df['texto_limpio'].tolist(), show_progress_bar=True)

    # UMAP: Reducción topológica a 2D para visualización
    print("  [FASE 2B] Reduciendo dimensiones espaciales con UMAP...")
    n_neighbors = min(15, len(df) - 1) if len(df) > 2 else 2
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, metric='cosine', random_state=42, n_jobs=1)
    coords = reducer.fit_transform(embeddings)
    
    df_copy = df.copy()
    df_copy['x'] = coords[:, 0]
    df_copy['y'] = coords[:, 1]
    df_copy['embedding'] = list(embeddings)

    # HDBSCAN: Detección espacial por densidad (Ruido = cluster -1)
    print("  [FASE 2C] Calculando densidades y agrupando ruido con HDBSCAN...")
    min_cluster = min(3, len(df) // 2) if len(df) > 5 else 2
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster, metric='euclidean', core_dist_n_jobs=1)
    
    # === CORRECCIÓN CLAVE ===
    # Alimentamos HDBSCAN con 'coords' (2D) en lugar de 'embeddings' (384D)
    df_copy['cluster_hdbscan'] = clusterer.fit_predict(coords)

    df_normal = df_copy[df_copy['cluster_hdbscan'] != -1].copy()
    df_outliers = df_copy[df_copy['cluster_hdbscan'] == -1].copy()

    # === MECANISMO ANTICOLAPSO ===
    # Si la dispersión es extrema y todo se etiquetó como ruido, forzamos la retención de los datos.
    if df_normal.empty and not df_copy.empty:
        print("  [Alerta] Dispersión extrema: HDBSCAN no detectó clústeres densos. Reteniendo dataset completo.")
        df_copy['cluster_hdbscan'] = 0  # Forzamos un clúster base
        df_normal = df_copy.copy()
        df_outliers = pd.DataFrame()

    # Extracción de N-gramas para el ruido residual
    if not df_outliers.empty:
        textos_outliers = df_outliers['comentario_grafo'].tolist()
        for n, label in zip([1, 2, 3], ['unigramas', 'bigramas', 'trigramas']):
            try:
                cv = CountVectorizer(ngram_range=(n, n), max_features=5)
                cv.fit(textos_outliers)
                df_outliers[f'top_{label}'] = ", ".join(cv.get_feature_names_out())
            except ValueError:
                df_outliers[f'top_{label}'] = "Insuficientes datos"

    return df_normal, df_outliers

def clasificar_sentimientos(df: pd.DataFrame, columna_texto: str) -> pd.DataFrame:
    """
    Clasificación robusta usando DistilBERT Multilingüe local.
    Optimizada con Procesamiento por Lotes (Batching) y Deduplicación.
    """
    if df.empty:
        df['sentimiento'] = None
        return df

    # Inicialización del modelo (device=-1 fuerza uso de CPU)
    classifier = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", device=-1)
    
    # 1. DEDUPLICACIÓN: Extraer solo los comentarios únicos para no procesar repetidos
    textos_unicos = df[columna_texto].astype(str).unique().tolist()
    print(f"  [Optimización] Se procesarán {len(textos_unicos)} textos únicos de un total de {len(df)}.")
    
    # 2. BATCHING: Pasar la lista completa al pipeline con batch_size=16
    # El pipeline maneja la paralelización interna automáticamente
    predicciones_unicas = []
    generador_pipeline = classifier(textos_unicos, truncation=True, max_length=512, batch_size=16)
    
    for res in tqdm(generador_pipeline, total=len(textos_unicos), desc="Clasificando Sentimientos", unit="lote"):
        predicciones_unicas.append(res)
        
    # 3. MAPEADO: Crear un diccionario con los resultados y aplicarlo al dataset original
    mapa_resultados = {texto: res['label'] for texto, res in zip(textos_unicos, predicciones_unicas)}
    mapping = {'positive': 'positivo', 'negative': 'negativo', 'neutral': 'negativo'}
    
    df_copy = df.copy()
    # Asignar el sentimiento a cada fila buscando en nuestro diccionario en memoria (operación instantánea)
    df_copy['sentimiento'] = df_copy[columna_texto].astype(str).map(lambda x: mapping.get(mapa_resultados[x], 'negativo'))
    
    return df_copy

def modelar_topicos(df: pd.DataFrame, columna_texto: str, umbral: int = 5) -> Dict[str, dict]:
    """
    Agrupa comentarios por sentimiento y extrae tópicos mediante los clusters de HDBSCAN.
    """
    resultados = {'positivo': {}, 'negativo': {}}
    
    for sent in ['positivo', 'negativo']:
        df_sent = df[df['sentimiento'] == sent]
        
        if len(df_sent) < umbral:
            if not df_sent.empty:
                cv = CountVectorizer(max_features=5)
                cv.fit(df_sent['texto_limpio'])
                resultados[sent] = {
                    'metodo': 'frecuencia',
                    'palabras_clave': list(cv.get_feature_names_out()),
                    'comentario_representativo': df_sent[columna_texto].iloc[0]
                }
        else:
            topicos_dict = {}
            for c_id in df_sent['cluster_hdbscan'].unique():
                df_cluster = df_sent[df_sent['cluster_hdbscan'] == c_id]
                
                cv = CountVectorizer(max_features=3)
                cv.fit(df_cluster['texto_limpio'])
                
                embs = np.vstack(df_cluster['embedding'].values)
                centroide = embs.mean(axis=0)
                distancias = [np.linalg.norm(emb - centroide) for emb in embs]
                
                topicos_dict[f'topico_{c_id}'] = {
                    'palabras_clave': list(cv.get_feature_names_out()),
                    'comentario_representativo': df_cluster[columna_texto].iloc[np.argmin(distancias)]
                }
            resultados[sent] = {'metodo': 'hdbscan', 'topicos': topicos_dict}
            
    return resultados

def extraer_red_semantica(df: pd.DataFrame) -> dict:
    """
    Extrae la matriz de co-ocurrencia para generar un grafo semántico.
    """
    if df.empty:
        return {"nodos": [], "enlaces": []}

    vectorizer = CountVectorizer(max_features=30, min_df=2)
    try:
        X = vectorizer.fit_transform(df['texto_limpio'].dropna())
    except ValueError:
        return {"nodos": [], "enlaces": []}

    palabras = vectorizer.get_feature_names_out()
    co_ocurrencia = (X.T * X).toarray()
    np.fill_diagonal(co_ocurrencia, 0)

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
                enlaces.append({"fuente": nodos[i], "destino": nodos[j], "peso": int(peso)})

    return {"nodos": nodos, "enlaces": enlaces}

def analizar_similitud_precio(df: pd.DataFrame, columna_texto: str) -> pd.DataFrame:
    """
    OBLIGATORIO: Análisis de costo/precio mediante similitud del coseno.
    """
    if df.empty:
        df['similitud_precio_costo'] = None
        return df

    encoder = get_encoder()
    embedding_concepto = encoder.encode(["precio valor costo"])
    embeddings_textos = np.vstack(df['embedding'].values)
    
    similitudes = cosine_similarity(embeddings_textos, embedding_concepto).flatten()
    
    df_copy = df.copy()
    df_copy['similitud_precio_costo'] = similitudes
    return df_copy.drop(columns=['embedding'])