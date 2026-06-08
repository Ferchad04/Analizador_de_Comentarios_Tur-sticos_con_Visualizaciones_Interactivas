# =============================================================================
# src/preprocesamiento.py
# Motor de limpieza textual NLP.
#
# CONTRATO DE COLUMNAS:
#   - comentario       -> texto original para visualización
#   - comentario_nlp   -> texto stemmed para PCA/HDBSCAN
#   - comentario_grafo -> texto limpio SIN stemming para NetworkX
# =============================================================================

import re
import unicodedata
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# Recursos NLTK
for recurso in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.download(recurso, quiet=True)
    except Exception:
        pass

# Configuración
_STEMMER_ES = SnowballStemmer("spanish")

_STOP_WORDS = (
    set(stopwords.words("spanish"))
    | set(stopwords.words("english"))
)

_MIN_TOKEN_LEN = 3

# Utilidades

def _normalizar_unicode(texto: str) -> str:
    return (
        unicodedata.normalize("NFD", texto)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )


def _limpiar_para_nlp(texto: str) -> str:

    texto = texto.lower()

    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"@\w+|#\w+", " ", texto)

    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)

    texto = re.sub(r"\s+", " ", texto).strip()

    texto = _normalizar_unicode(texto)

    return texto


def _limpiar_para_grafo(texto: str) -> str:

    texto = texto.lower()

    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"@\w+|#\w+", " ", texto)

    # conserva acentos y ñ
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)

    texto = re.sub(r"\s+", " ", texto).strip()

    return texto


def _tokenizar_y_filtrar(texto: str) -> list[str]:

    tokens = word_tokenize(texto, language="spanish")

    return [
        token
        for token in tokens
        if token not in _STOP_WORDS
        and len(token) >= _MIN_TOKEN_LEN
    ]


def _procesar_texto_nlp(texto: str) -> str:

    texto_limpio = _limpiar_para_nlp(str(texto))

    tokens = _tokenizar_y_filtrar(texto_limpio)

    stems = [_STEMMER_ES.stem(t) for t in tokens]

    resultado = " ".join(stems)

    return resultado if resultado.strip() else "__vacio__"


def _procesar_texto_grafo(texto: str) -> str:

    texto_limpio = _limpiar_para_grafo(str(texto))

    tokens = _tokenizar_y_filtrar(texto_limpio)

    resultado = " ".join(tokens)

    return resultado if resultado.strip() else "__vacio__"


# API pública

def preprocesar(
    df: pd.DataFrame,
    columna_visual: str = "comentario"
):

    if columna_visual not in df.columns:
        raise ValueError(
            f"Columna '{columna_visual}' no encontrada. "
            f"Disponibles: {list(df.columns)}"
        )

    df_trabajo = df.copy()

    # DEDUPLICACIÓN
    total_antes = len(df_trabajo)

    mascara_dup = df_trabajo.duplicated(
        subset=[columna_visual],
        keep="first"
    )

    df_duplicados = df_trabajo[mascara_dup].copy()

    df_trabajo = (
        df_trabajo[~mascara_dup]
        .reset_index(drop=True)
    )

    n_dup = total_antes - len(df_trabajo)

    if n_dup > 0:
        print(
            f"  [DEDUP] {n_dup} duplicado(s) eliminado(s). "
            f"Únicos restantes: {len(df_trabajo)}"
        )

    # COLUMNAS NLP
    df_trabajo["comentario_nlp"] = (
        df_trabajo[columna_visual].copy()
    )

    df_trabajo["comentario_grafo"] = (
        df_trabajo[columna_visual].copy()
    )

    # FILTRO DE NULOS
    mascara_nulos = (
        df_trabajo[columna_visual]
        .astype(str)
        .str.strip()
        .isin(["", "nan", "None"])
    )

    df_outliers_nulos = (
        df_trabajo[mascara_nulos]
        .copy()
    )

    df_trabajo = (
        df_trabajo[~mascara_nulos]
        .reset_index(drop=True)
    )

    # PROCESAMIENTO NLP
    print(
        f"  [NLP] Procesando "
        f"{len(df_trabajo)} registros únicos..."
    )

    df_trabajo["comentario_nlp"] = (
        df_trabajo["comentario_nlp"]
        .apply(_procesar_texto_nlp)
    )

    df_trabajo["comentario_grafo"] = (
        df_trabajo["comentario_grafo"]
        .apply(_procesar_texto_grafo)
    )

    # VACÍOS NLP
    mascara_vacios = (
        df_trabajo["comentario_nlp"] == "__vacio__"
    )

    df_outliers_nlp = (
        df_trabajo[mascara_vacios]
        .copy()
    )

    df_clasificado = (
        df_trabajo[~mascara_vacios]
        .reset_index(drop=True)
    )

    # CONSOLIDAR OUTLIERS
    df_outliers = pd.concat(
        [
            df_duplicados,
            df_outliers_nulos,
            df_outliers_nlp,
        ],
        ignore_index=True,
    )

    print(
        f"  [RESULT] Para modelado: {len(df_clasificado)} | "
        f"Outliers totales: {len(df_outliers)} "
        f"(dup={n_dup}, "
        f"nulos={len(df_outliers_nulos)}, "
        f"vacíos_nlp={len(df_outliers_nlp)})"
    )

    assert "comentario" in df_clasificado.columns
    assert "comentario_nlp" in df_clasificado.columns
    assert "comentario_grafo" in df_clasificado.columns

    return df_clasificado, df_outliers