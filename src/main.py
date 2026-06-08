import argparse
import sys
import gc

import preprocesamiento
import modelado
import visualizacion

def main():
    # Configuración estricta de los 5 parámetros requeridos por los lineamientos
    parser = argparse.ArgumentParser(description="Analizador de Comentarios Turísticos - Alta Eficiencia")
    parser.add_argument("ruta_datos", type=str, help="Ruta del archivo CSV o Parquet")
    parser.add_argument("columna_texto", type=str, help="Nombre de la columna con los comentarios")
    parser.add_argument("idioma", type=str, choices=["es", "en", "fr"], help="Idioma objetivo (es, en, fr)")
    parser.add_argument("titulo_reporte", type=str, help="Título del reporte generado")
    parser.add_argument("paleta_colores", type=str, choices=["viridis", "cividis", "plasma", "inferno"], help="Paleta accesible")

    args = parser.parse_args()
    print(f"\n=== Iniciando Pipeline Optimizado: {args.titulo_reporte} ===")

    try:
        # FASE 1: Ingesta y limpieza en memoria RAM (Soporta CSV y Parquet)
        print("[1/3] Ejecutando limpieza y preprocesamiento...")
        df_original = preprocesamiento.cargar_datos(args.ruta_datos, args.columna_texto)
        df_limpio = preprocesamiento.procesar_nlp(df_original, args.columna_texto, args.idioma)
        
        # Purgar datos originales crudos de la RAM para evitar saturación
        del df_original
        gc.collect()

        # --- OPTIMIZACIÓN DE ESCALABILIDAD: MUESTREO REPRESENTATIVO ---
        # Evita el colapso de memoria por HDBSCAN y la congelación de Plotly en el navegador
        LIMITE_PUNTOS = 6000
        if len(df_limpio) > LIMITE_PUNTOS:
            print(f"  [Muestreo] Dataset masivo detectado ({len(df_limpio)} filas).")
            print(f"  [Muestreo] Extrayendo muestra aleatoria representativa de {LIMITE_PUNTOS} reseñas para estabilidad.")
            df_limpio = df_limpio.sample(n=LIMITE_PUNTOS, random_state=42).reset_index(drop=True)
            gc.collect()
        # ---------------------------------------------------------------

        # FASE 2: Núcleo Matemático (Paralelizado, Mono-núcleo seguro y Batching)
        print("[2/3] Ejecutando modelado espacial HDBSCAN y análisis semántico...")
        df_normal, df_outliers = modelado.detectar_outliers_y_ngramas(df_limpio, args.columna_texto)
        
        # Purgar el DataFrame intermedio previo al clustering
        del df_limpio
        gc.collect()

        df_clasificado = modelado.clasificar_sentimientos(df_normal, args.columna_texto)
        resultados_topicos = modelado.modelar_topicos(df_clasificado, args.columna_texto)
        resultados_similitud = modelado.analizar_similitud_precio(df_clasificado, args.columna_texto)

        # FASE 3: Exportación Visual y Reportes Off-line
        print("[3/3] Generando visualizaciones interactivas...")
        visualizacion.generar_reporte(
            df_clasificado, 
            df_outliers,
            resultados_topicos, 
            resultados_similitud, 
            args.titulo_reporte, 
            args.paleta_colores
        )

        # Limpieza absoluta de tensores, listas densas y dataframes antes de concluir
        del df_clasificado, df_outliers, resultados_topicos, resultados_similitud
        gc.collect()

        print("\n=== Pipeline finalizado exitosamente ===")

    except Exception as e:
        print(f"\nError crítico durante la ejecución del orquestador: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()