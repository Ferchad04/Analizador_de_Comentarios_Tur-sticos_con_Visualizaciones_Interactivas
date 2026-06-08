# =============================================================================
# src/etl_estandarizacion.py
# Script de uso único (one-shot). Normaliza los 5 CSVs fuente al esquema
# canónico del pipeline: columna 'comentario' en UTF-8 limpio.
# Ejecutar desde la raíz del proyecto: python src/etl_estandarizacion.py
# =============================================================================

import os
import pandas as pd

# ---------------------------------------------------------------------------
# MAPA DE INGESTA
# Cada entrada define: archivo fuente, columna objetivo en ese archivo y
# el nombre base del archivo de salida limpio.
# Si la codificación primaria falla, el ETL intentará el fallback 'latin-1'.
# ---------------------------------------------------------------------------
MAPA_DATASETS = [
    {
        "archivo_origen": "data/huatulco.csv",
        "columna_origen": "Comentarios",   # Tiene corrupción ISO/ANSI → ''
        "encoding_primario": "utf-8",
        "encoding_fallback": "latin-1",
        "archivo_destino": "data/huatulco_limpio.csv",
    },
    {
        "archivo_origen": "data/la_paz.csv",
        "columna_origen": "texto",
        "encoding_primario": "utf-8",
        "encoding_fallback": "latin-1",
        "archivo_destino": "data/la_paz_limpio.csv",
    },
    {
        "archivo_origen": "data/riviera_maya.csv",
        "columna_origen": "Comentario",
        "encoding_primario": "utf-8",
        "encoding_fallback": "latin-1",
        "archivo_destino": "data/riviera_maya_limpio.csv",
    },
    {
        "archivo_origen": "data/riviera_nayarit.csv",
        "columna_origen": "Comentario",    # Encoding corrupto + cols. nulas al final
        "encoding_primario": "utf-8",
        "encoding_fallback": "latin-1",
        "archivo_destino": "data/riviera_nayarit_limpio.csv",
    },
    {
        "archivo_origen": "data/puerto_vallarta.csv",
        "columna_origen": "review_text",
        "encoding_primario": "utf-8",
        "encoding_fallback": "latin-1",
        "archivo_destino": "data/puerto_vallarta_limpio.csv",
    },
]


def _leer_csv_robusto(ruta: str, encoding_primario: str, encoding_fallback: str) -> pd.DataFrame:
    """
    Intenta leer el CSV con el encoding declarado. Si lanza UnicodeDecodeError,
    reintenta con el fallback 'latin-1' usando errors='replace' para sustituir
    cualquier carácter no mapeable por el símbolo de reemplazo Unicode (U+FFFD)
    en vez de abortar la carga.
    """
    try:
        df = pd.read_csv(ruta, encoding=encoding_primario)
        print(f"  [OK] Leído con {encoding_primario}: {ruta}")
        return df
    except UnicodeDecodeError:
        print(f"  [WARN] Encoding '{encoding_primario}' falló. Reintentando con '{encoding_fallback}'...")
        df = pd.read_csv(ruta, encoding=encoding_fallback, encoding_errors="replace")
        print(f"  [OK] Leído con {encoding_fallback} (errors=replace): {ruta}")
        return df


def _limpiar_columnas_nulas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas generadas automáticamente por pandas cuando el CSV tiene
    comas sobrantes al final de cada fila.
    Se identifica cualquier columna cuyo nombre empiece con 'Unnamed'.
    """
    cols_basura = [c for c in df.columns if str(c).startswith("Unnamed")]
    if cols_basura:
        print(f"  [CLEAN] Eliminando {len(cols_basura)} columna(s) nula(s): {cols_basura}")
        df = df.drop(columns=cols_basura)
    return df


def _extraer_y_renombrar(df: pd.DataFrame, columna_origen: str, archivo: str) -> pd.DataFrame:
    """
    Extrae únicamente la columna de texto útil y la renombra al nombre canónico
    'comentario'. Descarta todas las demás columnas del dataset original para
    garantizar un esquema uniforme aguas abajo.
    Lanza KeyError con mensaje descriptivo si la columna no existe.
    """
    if columna_origen not in df.columns:
        disponibles = list(df.columns)
        raise KeyError(
            f"[ERROR] Columna '{columna_origen}' no encontrada en '{archivo}'. "
            f"Columnas disponibles: {disponibles}"
        )
    # Conservar únicamente la columna objetivo → renombrar → schema canónico
    df_limpio = df[[columna_origen]].copy()
    df_limpio.rename(columns={columna_origen: "comentario"}, inplace=True)
    return df_limpio


def _sanitizar_texto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica saneamiento básico sobre la columna 'comentario':
      1. Convierte a string para evitar celdas numéricas/NaN flotando.
      2. Elimina espacios en blanco extremos (strip).
      3. Descarta filas completamente vacías o con el literal 'nan'.
      4. Fuerza re-encoding UTF-8 carácter a carácter descartando bytes
         inválidos residuales (encode → decode con errors='ignore').
    """
    df["comentario"] = df["comentario"].astype(str).str.strip()

    # Re-encoding limpio: elimina bytes corruptos residuales sin levantar excepción
    df["comentario"] = df["comentario"].apply(
        lambda x: x.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    )

    # Filtrar filas vacías o el literal 'nan' que queda al castear NaN → str
    mascara_valida = (df["comentario"] != "") & (df["comentario"].str.lower() != "nan")
    descartadas = (~mascara_valida).sum()
    if descartadas > 0:
        print(f"  [DROP] {descartadas} fila(s) vacía(s) o nulas descartadas.")
    df = df[mascara_valida].reset_index(drop=True)
    return df


def ejecutar_etl() -> None:
    """
    Punto de entrada principal del ETL. Itera sobre MAPA_DATASETS ejecutando
    el pipeline de extracción, limpieza y guardado para cada dataset.
    """
    print("=" * 65)
    print("  ETL DE ESTANDARIZACIÓN — PIPELINE TURÍSTICO NLP")
    print("=" * 65)

    for config in MAPA_DATASETS:
        archivo_origen = config["archivo_origen"]
        columna_origen = config["columna_origen"]
        archivo_destino = config["archivo_destino"]

        print(f"\n▶ Procesando: {archivo_origen}")
        print(f"  Columna objetivo: '{columna_origen}'")

        # verificar existencia del archivo fuente
        if not os.path.exists(archivo_origen):
            print(f"  [SKIP] Archivo no encontrado: {archivo_origen}")
            continue

        try:
            # Paso 1: Carga robusta con fallback de encoding
            df_raw = _leer_csv_robusto(
                archivo_origen,
                config["encoding_primario"],
                config["encoding_fallback"],
            )

            # Paso 2: Purgar columnas 'Unnamed'
            df_raw = _limpiar_columnas_nulas(df_raw)

            # Paso 3: Extraer columna objetivo y renombrar a schema canónico
            df_canonico = _extraer_y_renombrar(df_raw, columna_origen, archivo_origen)

            # Paso 4: Saneamiento de texto (encoding residual, filas vacías)
            df_canonico = _sanitizar_texto(df_canonico)

            # Paso 5: Validación de volumen mínimo exigido por la rúbrica
            total_registros = len(df_canonico)
            print(f"  [INFO] Registros válidos tras limpieza: {total_registros}")
            if total_registros < 500:
                print(
                    f"  [WARN] El dataset tiene {total_registros} registros. "
                    f"La rúbrica exige mínimo 500 por destino."
                )

            # Paso 6: Persistir CSV limpio en UTF-8 sin índice
            df_canonico.to_csv(archivo_destino, index=False, encoding="utf-8")
            print(f"  [SAVED] → {archivo_destino} ({total_registros} filas, 1 columna: 'comentario')")

        except KeyError as e:
            # Error de columna no encontrada: logear y continuar con el resto
            print(str(e))
        except Exception as e:
            print(f"  [ERROR] Fallo inesperado en '{archivo_origen}': {e}")

    print("\n" + "=" * 65)
    print("  ETL completado. Archivos listos en data/*_limpio.csv")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ejecutar_etl()
