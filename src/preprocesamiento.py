import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import sys
import os
import gc

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def _leer_datos_robusto(ruta: str) -> pd.DataFrame:
    """
    Detecta la extensión del archivo. Lee nativamente Parquet o aplica
    el fallback robusto de codificación para archivos CSV.
    """
    ext = os.path.splitext(ruta)[1].lower()
    try:
        if ext == '.parquet':
            return pd.read_parquet(ruta)
        else:
            return pd.read_csv(ruta, encoding="utf-8")
    except UnicodeDecodeError:
        print(f" [WARN] Fallo de codificación UTF-8. Reintentando con latin-1: {ruta}", file=sys.stderr)
        return pd.read_csv(ruta, encoding="latin-1", encoding_errors="replace")
    except FileNotFoundError:
        print(f" [ERROR] No se encontró el archivo: {ruta}", file=sys.stderr)
        sys.exit(1)

def _sanitizar_texto(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    df[columna] = df[columna].astype(str).str.strip()
    df[columna] = df[columna].apply(
        lambda x: x.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    )
    mascara_valida = (df[columna] != "") & (df[columna].str.lower() != "nan")
    return df[mascara_valida].reset_index(drop=True)

def cargar_datos(ruta_csv: str, columna_texto: str) -> pd.DataFrame:
    df = _leer_datos_robusto(ruta_csv)
    
    cols_basura = [c for c in df.columns if str(c).startswith("Unnamed")]
    if cols_basura:
        df = df.drop(columns=cols_basura)
        
    if columna_texto not in df.columns:
        print(f" [ERROR] La columna '{columna_texto}' no existe en el archivo.", file=sys.stderr)
        sys.exit(1)
        
    df = df.dropna(subset=[columna_texto]).copy()
    df = _sanitizar_texto(df, columna_texto)
    return df

def limpiar_texto(texto: str) -> str:
    texto = str(texto).lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'\d+', '', texto)
    return re.sub(r'[^\w\s]', '', texto).strip()

def procesar_nlp(df: pd.DataFrame, columna_texto: str, idioma: str) -> pd.DataFrame:
    mapa_idiomas = {'es': 'spanish', 'en': 'english', 'fr': 'french'}
    idioma_nltk = mapa_idiomas.get(idioma)
    
    if not idioma_nltk:
        print(f" [ERROR] Idioma '{idioma}' no soportado.", file=sys.stderr)
        sys.exit(1)

    stop_words = set(stopwords.words(idioma_nltk))
    stemmer = SnowballStemmer(idioma_nltk)

    # 1. Pipeline con Stemming (Para matemáticas: HDBSCAN, UMAP)
    def pipeline_texto(texto: str) -> str:
        tokens = limpiar_texto(texto).split()
        return " ".join([stemmer.stem(w) for w in tokens if w not in stop_words])

    # 2. Pipeline sin Stemming (Para visualización legible: N-gramas)
    def pipeline_grafo(texto: str) -> str:
        tokens = limpiar_texto(texto).split()
        return " ".join([w for w in tokens if w not in stop_words])

    df_copy = df.copy()
    df_copy['texto_limpio'] = df_copy[columna_texto].apply(pipeline_texto)
    df_copy['comentario_grafo'] = df_copy[columna_texto].apply(pipeline_grafo)
    
    # Liberación explícita de RAM
    gc.collect()
    return df_copy