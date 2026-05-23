# src/preprocesamiento.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import sys

# Asegurar disponibilidad de recursos locales (se descargan solo si no existen)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def cargar_datos(ruta_csv: str, columna_texto: str) -> pd.DataFrame:
    """
    Carga el dataset y valida la existencia de la columna objetivo.
    """
    try:
        df = pd.read_csv(ruta_csv)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta {ruta_csv}", file=sys.stderr)
        sys.exit(1)
        
    if columna_texto not in df.columns:
        print(f"Error: La columna '{columna_texto}' no existe en el CSV.", file=sys.stderr)
        sys.exit(1)
        
    # Eliminar filas donde el texto sea nulo
    df = df.dropna(subset=[columna_texto]).copy()
    return df

def limpiar_texto(texto: str) -> str:
    """
    Aplica expresiones regulares para eliminar ruido del texto.
    """
    texto = str(texto).lower()
    # Eliminar URLs
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    # Eliminar números (asumiendo que no son relevantes para el análisis general)
    texto = re.sub(r'\d+', '', texto)
    # Eliminar caracteres especiales y puntuación
    texto = re.sub(r'[^\w\s]', '', texto)
    
    return texto.strip()

def procesar_nlp(df: pd.DataFrame, columna_texto: str, idioma: str) -> pd.DataFrame:
    """
    Aplica tokenización, remoción de stopwords y stemming basado en el idioma.
    Retorna el DataFrame con el texto original reemplazado por la versión procesada.
    """
    # Mapeo de códigos de idioma de la línea de comandos a NLTK
    mapa_idiomas = {
        'es': 'spanish',
        'en': 'english',
        'fr': 'french'
    }
    
    idioma_nltk = mapa_idiomas.get(idioma)
    if not idioma_nltk:
        print(f"Error: Idioma '{idioma}' no soportado. Usa es, en o fr.", file=sys.stderr)
        sys.exit(1)

    stop_words = set(stopwords.words(idioma_nltk))
    stemmer = SnowballStemmer(idioma_nltk)

    def pipeline_texto(texto: str) -> str:
        texto_limpio = limpiar_texto(texto)
        # Tokenización básica por espacios
        tokens = texto_limpio.split()
        # Filtrado y stemming
        tokens_procesados = [
            stemmer.stem(word) for word in tokens if word not in stop_words
        ]
        return " ".join(tokens_procesados)

    df_copy = df.copy()
    # Se sobrescribe la columna o se crea una nueva para el modelado posterior
    df_copy[columna_texto] = df_copy[columna_texto].apply(pipeline_texto)
    
    return df_copy