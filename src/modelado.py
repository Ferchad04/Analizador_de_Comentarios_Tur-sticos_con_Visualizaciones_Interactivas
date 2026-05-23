# src/modelado.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from typing import Tuple, Dict

def detectar_outliers_y_ngramas(df: pd.DataFrame, columna_texto: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detecta comentarios atípicos usando Isolation Forest y extrae sus N-gramas.
    Los comentarios normales se retornan por separado.
    """
    if df.empty:
        return df, pd.DataFrame()

    # Vectorización temporal para detección de anomalías numéricas
    tfidf = TfidfVectorizer(max_features=500)
    x_tfidf = tfidf.fit_transform(df[columna_texto])
    
    # Modelo Isolation Forest (Contaminación estimada al 10%)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    preds = iso_forest.fit_predict(x_tfidf.toarray())
    
    df_copy = df.copy()
    df_copy['outlier'] = preds
    
    df_normal = df_copy[df_copy['outlier'] == 1].drop(columns=['outlier'])
    df_outliers = df_copy[df_copy['outlier'] == -1].copy()
    
    if not df_outliers.empty:
        # Extracción de N-gramas (Unigramas, Bigramas, Trigramas)
        comentarios_outliers = df_outliers[columna_texto].tolist()
        
        for n, label in zip([1, 2, 3], ['unigramas', 'bigramas', 'trigramas']):
            try:
                cv = CountVectorizer(ngram_range=(n, n), max_features=5)
                cv_matrix = cv.fit_transform(comentarios_outliers)
                vocab = cv.get_feature_names_out()
                df_outliers[f'top_{label}'] = ", ".join(vocab)
            except ValueError:
                df_outliers[f'top_{label}'] = "Insuficientes datos"
                
    return df_normal, df_outliers

def clasificar_sentimientos(df: pd.DataFrame, columna_texto: str) -> pd.DataFrame:
    """
    Clasifica los comentarios en positivos y negativos usando un modelo Transformer local.
    """
    if df.empty:
        df['sentimiento'] = None
        return df

    # Modelo multilingüe ligero descargado localmente
    classifier = pipeline(
        "sentiment-analysis", 
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        device=-1 # Ejecución en CPU para portabilidad absoluta
    )
    
    textos = df[columna_texto].astype(str).tolist()
    resultados = classifier(textos)
    
    # Homologar etiquetas del modelo a los grupos requeridos
    mapping = {'positive': 'positivo', 'negative': 'negativo', 'neutral': 'negativo'}
    df_copy = df.copy()
    df_copy['sentimiento'] = [mapping.get(res['label'], 'negativo') for res in resultados]
    
    return df_copy

def modelar_topicos(df: pd.DataFrame, columna_texto: str, umbral: int = 5) -> Dict[str, dict]:
    """
    Aplica modelado de tópicos por densidad. Si el grupo es menor al umbral,
    utiliza frecuencias de palabras absolutas.
    """
    resultados = {'positivo': {}, 'negativo': {}}
    
    for sent in ['positivo', 'negativo']:
        df_sent = df[df['sentimiento'] == sent]
        
        if len(df_sent) < umbral:
            # Enfoque por frecuencia de palabras (conteo directo)
            if not df_sent.empty:
                cv = CountVectorizer(max_features=5)
                cv.fit(df_sent[columna_texto])
                resultados[sent] = {
                    'metodo': 'frecuencia',
                    'palabras_clave': list(cv.get_feature_names_out()),
                    'comentario_representativo': df_sent[columna_texto].iloc[0]
                }
        else:
            # Enfoque basado en embeddings para tópicos locales (K-Means como alternativa ligera offline)
            from sklearn.cluster import KMeans
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = model.encode(df_sent[columna_texto].tolist())
            
            n_clusters = min(3, len(df_sent))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
            
            # Identificar el comentario más cercano al centroide de cada cluster
            comentarios = df_sent[columna_texto].tolist()
            topicos_dict = {}
            
            for idx in range(n_clusters):
                centroide = kmeans.cluster_centers_[idx]
                indices_cluster = np.where(kmeans.labels_ == idx)[0]
                
                # Calcular distancias euclidianas al centroide dentro del cluster
                distancias = [np.linalg.norm(embeddings[i] - centroide) for i in indices_cluster]
                idx_mas_cercano = indices_cluster[np.argmin(distancias)]
                
                # Extraer palabras clave representativas del subgrupo
                sub_textos = [comentarios[i] for i in indices_cluster]
                cv = CountVectorizer(max_features=3)
                cv.fit(sub_textos)
                
                topicos_dict[f'topico_{idx}'] = {
                    'palabras_clave': list(cv.get_feature_names_out()),
                    'comentario_representativo': comentarios[idx_mas_cercano]
                }
                
            resultados[sent] = {
                'metodo': 'clustering_semantico',
                'topicos': topicos_dict
            }
            
    return resultados

def analizar_similitud_precio(df: pd.DataFrame, columna_texto: str) -> pd.DataFrame:
    """
    Calcula la similitud semántica mediante Embeddings con el concepto "precio / valor / costo".
    """
    if df.empty:
        df['similitud_precio_costo'] = None
        return df

    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generar embeddings de los textos y del concepto sintético objetivo
    embeddings_textos = model.encode(df[columna_texto].tolist())
    embedding_concepto = model.encode(["precio valor costo"])
    
    # Cálculo de similitud del coseno
    similitudes = cosine_similarity(embeddings_textos, embedding_concepto).flatten()
    
    df_copy = df.copy()
    df_copy['similitud_precio_costo'] = similitudes
    
    return df_copy