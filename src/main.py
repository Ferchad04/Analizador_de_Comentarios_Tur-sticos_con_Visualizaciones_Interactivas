# =============================================================================
# src/main.py
# Orquestador Batch del Pipeline NLP Turístico.
# Detecta automáticamente todos los archivos *_limpio.csv en data/ y ejecuta
# el pipeline completo (preprocesamiento → modelado → visualización) por cada
# uno, generando un dashboard HTML por destino turístico en una sola corrida.
#
# Uso:
#   python src/main.py
#   python src/main.py --data-dir data/ --output-dir .   (flags opcionales)
#
# Prerequisitos:
#   1. Haber ejecutado src/etl_estandarizacion.py primero.
#   2. Tener modelado.py y visualizacion.py en src/ (sin modificaciones).
# =============================================================================

import os
import sys
import glob
import argparse
import traceback
from datetime import datetime

import pandas as pd

# Ajuste de path: permite importar módulos hermanos desde src/ sin instalar
# el paquete. Necesario cuando se ejecuta como script directo.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Importaciones de módulos propios del pipeline
from preprocesamiento import preprocesar
import modelado          # API: modelado.ejecutar(df_clasificado) → datos_modelo
import visualizacion     # API: visualizacion.generar_reporte(...) → ruta_html


# Constantes de configuración (sobreescribibles con flags CLI)
DEFAULT_DATA_DIR   = "data"         # Directorio que contiene los *_limpio.csv
DEFAULT_OUTPUT_DIR = "."            # Directorio raíz donde se guardan los HTML
PATRON_ARCHIVOS    = "*_limpio.csv" # Glob pattern para descubrimiento automático
COLUMNA_VISUAL     = "comentario"   # Columna canónica de texto crudo (ETL garantiza esto)
MIN_REGISTROS      = 500            # Mínimo exigido por rúbrica del profesor


# Parseo de argumentos CLI 
def _parsear_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline NLP Batch — Análisis de Destinos Turísticos"
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directorio con los CSVs limpios (default: '{DEFAULT_DATA_DIR}')",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directorio de salida para los HTML (default: '{DEFAULT_OUTPUT_DIR}')",
    )
    return parser.parse_args()


# Utilidades internas

def _descubrir_archivos(data_dir: str) -> list[str]:
    """
    Usa glob para encontrar todos los CSVs con sufijo '_limpio.csv' en data_dir.
    Retorna lista ordenada de rutas absolutas.
    Aborta si no encuentra ninguno.
    """
    patron = os.path.join(data_dir, PATRON_ARCHIVOS)
    archivos = sorted(glob.glob(patron))

    if not archivos:
        print(
            f"\n[FATAL] No se encontraron archivos con patrón '{patron}'.\n"
            f"  Asegúrate de haber ejecutado primero: python src/etl_estandarizacion.py\n"
        )
        sys.exit(1)

    print(f"\n[DISCOVERY] {len(archivos)} dataset(s) encontrado(s) en '{data_dir}':")
    for ruta in archivos:
        print(f"  • {os.path.basename(ruta)}")

    return archivos


def _extraer_nombre_destino(ruta_csv: str) -> str:
    """
    Deriva el nombre legible del destino a partir del nombre de archivo.
    Ejemplo: 'data/riviera_maya_limpio.csv' → 'Riviera Maya'
    """
    nombre_base = os.path.basename(ruta_csv)                     # 'riviera_maya_limpio.csv'
    sin_extension = os.path.splitext(nombre_base)[0]             # 'riviera_maya_limpio'
    sin_sufijo = sin_extension.replace("_limpio", "")            # 'riviera_maya'
    titulo = sin_sufijo.replace("_", " ").title()                # 'Riviera Maya'
    return titulo


def _cargar_csv(ruta: str) -> pd.DataFrame | None:
    """
    Carga el CSV limpio (UTF-8 garantizado por el ETL).
    Retorna None si el archivo no puede ser leído o está vacío.
    """
    try:
        df = pd.read_csv(ruta, encoding="utf-8")
        if df.empty:
            print(f"  [SKIP] Archivo vacío: {ruta}")
            return None
        return df
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar '{ruta}': {e}")
        return None


def _construir_ruta_salida(output_dir: str, nombre_destino: str) -> str:
    """
    Construye la ruta del dashboard HTML de salida con timestamp para
    evitar sobreescribir corridas anteriores.
    Ejemplo: './reporte_ejecutivo_Riviera_Maya_20250607_143022.html'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = (
        f"reporte_ejecutivo_{nombre_destino.replace(' ', '_')}_{timestamp}.html"
    )
    return os.path.join(output_dir, nombre_archivo)


# Pipeline por dataset (unidad de trabajo del batch)

def _ejecutar_pipeline_individual(
    ruta_csv: str,
    output_dir: str,
    indice: int,
    total: int,
) -> bool:
    """
    Ejecuta el pipeline completo para un único CSV limpio:
      1. Carga del DataFrame.
      2. Validación de volumen mínimo.
      3. Preprocesamiento (separación comentario / comentario_nlp).
      4. Modelado NLP (TF-IDF, SentenceTransformers en CPU, NetworkX, PCA, Coseno).
      5. Generación del dashboard HTML offline.

    Retorna True si el pipeline terminó exitosamente, False si fue abortado.
    El pipeline es tolerante a fallos individuales: un error en un dataset
    no aborta la corrida de los demás.
    """
    nombre_destino = _extraer_nombre_destino(ruta_csv)

    separador = "─" * 60
    print(f"\n{separador}")
    print(f"  [{indice}/{total}] INICIANDO: {nombre_destino}")
    print(f"  Fuente: {ruta_csv}")
    print(separador)

    # Paso 1: Carga
    df = _cargar_csv(ruta_csv)
    if df is None:
        return False

    # Paso 2: Validación de volumen mínimo (rúbrica: ≥ 500 reseñas)
    if len(df) < MIN_REGISTROS:
        print(
            f"  [WARN] Solo {len(df)} registros. La rúbrica exige {MIN_REGISTROS} mínimo. "
            f"Se continúa de todas formas."
        )

    # Paso 3: Preprocesamiento
    # Garantiza la separación entre 'comentario' (visual) y
    # 'comentario_nlp' (matemático) sin mutilar el texto original.
    print(f"  [FASE 1/3] Preprocesamiento...")
    df_clasificado, df_outliers = preprocesar(df, columna_visual=COLUMNA_VISUAL)

    if df_clasificado.empty:
        print(f"  [ABORT] Todos los registros fueron descartados como outliers. Saltando.")
        return False

    print(
        f"  → Registros para modelado: {len(df_clasificado)} | "
        f"Outliers/Ruido: {len(df_outliers)}"
    )

    # Paso 4: Modelado NLP (delegado a modelado.py)
    # Ejecuta en CPU (forzado por arquitectura del proyecto).
    # Lee/escribe caché joblib en cache_nlp/ automáticamente.
    print(f"  [FASE 2/3] Modelado NLP (CPU + caché joblib)...")
    datos_modelo = modelado.ejecutar(
        df_clasificado=df_clasificado,
        nombre_destino=nombre_destino,
    )

    # Paso 5: Generación del Dashboard HTML (delegado a visualizacion.py)
    # El parámetro columna_original="comentario" garantiza que la capa
    # visual lea el texto humano legible, no el stemmed de NLP.
    print(f"  [FASE 3/3] Generando Dashboard HTML offline...")
    ruta_html = _construir_ruta_salida(output_dir, nombre_destino)

    visualizacion.generar_reporte(
        df_clasificado=df_clasificado,
        df_outliers=df_outliers,
        datos_modelo=datos_modelo,
        titulo=f"Análisis NLP — {nombre_destino}",
        columna_original=COLUMNA_VISUAL,    
        ruta_salida=ruta_html,
    )

    print(f"  [✓] Dashboard generado: {ruta_html}")
    return True


# Orquestador Batch principal

def main() -> None:
    """
    Punto de entrada del batch. Descubre todos los CSVs limpios,
    itera sobre ellos y acumula un resumen de la corrida.
    """
    args = _parsear_args()

    print("\n" + "=" * 60)
    print("  PIPELINE NLP — ANÁLISIS TURÍSTICO BATCH")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Descubrimiento automático de datasets
    archivos_csv = _descubrir_archivos(args.data_dir)
    total = len(archivos_csv)

    # Contadores para el resumen final
    exitosos: list[str] = []
    fallidos: list[str] = []

    # BUCLE PRINCIPAL BATCH
    # Cada iteración es independiente.
    for indice, ruta_csv in enumerate(archivos_csv, start=1):
        nombre_destino = _extraer_nombre_destino(ruta_csv)
        try:
            ok = _ejecutar_pipeline_individual(
                ruta_csv=ruta_csv,
                output_dir=args.output_dir,
                indice=indice,
                total=total,
            )
            if ok:
                exitosos.append(nombre_destino)
            else:
                fallidos.append(nombre_destino)

        except Exception:
            # Captura cualquier excepción no controlada dentro del pipeline
            # individual, la registra con traceback completo y continúa.
            print(f"\n  [ERROR NO CONTROLADO] en '{nombre_destino}':")
            traceback.print_exc()
            fallidos.append(nombre_destino)

    # RESUMEN DE CORRIDA
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL DE CORRIDA BATCH")
    print("=" * 60)
    print(f"  Total datasets procesados : {total}")
    print(f"  Dashboards generados (✓)  : {len(exitosos)}")
    print(f"  Fallos / Saltados    (✗)  : {len(fallidos)}")

    if exitosos:
        print("\n  Exitosos:")
        for nombre in exitosos:
            print(f"    ✓ {nombre}")

    if fallidos:
        print("\n  Fallidos (revisar logs):")
        for nombre in fallidos:
            print(f"    ✗ {nombre}")

    print(f"\n  Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

    # Exit code: 0 si todos exitosos, 1 si alguno falló (útil para CI/scripts)
    sys.exit(0 if not fallidos else 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()