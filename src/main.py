# src/main.py
import argparse
import sys
import logging

# Importación de los módulos del equipo
import preprocesamiento
import modelado
import visualizacion

def configurar_cli():
    """Configura e inicializa la interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline NLP Offline - Analizador de Comentarios Turísticos",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Grupo 1: Configuración de Entrada (I/O)
    grupo_entrada = parser.add_argument_group('Configuración de Entrada')
    grupo_entrada.add_argument("--input", required=True, type=str, help="Ruta del archivo CSV.")
    grupo_entrada.add_argument("--columna", required=True, type=str, help="Columna que contiene los comentarios.")
    grupo_entrada.add_argument("--idioma", required=True, type=str, choices=["es", "en", "fr"], help="Idioma objetivo para tokenización y stopwords.")

    # Grupo 2: Personalización Visual
    grupo_visual = parser.add_argument_group('Personalización Visual')
    grupo_visual.add_argument("--titulo", required=True, type=str, help="Título del reporte HTML de salida.")
    grupo_visual.add_argument("--paleta", required=True, type=str, choices=["viridis", "cividis", "plasma", "inferno"], help="Paleta de colores accesible.")

    # Grupo 3: Ejecución y Depuración
    grupo_ejecucion = parser.add_argument_group('Opciones de Ejecución')
    grupo_ejecucion.add_argument("--verbose", action="store_true", help="Activa el registro detallado de operaciones.")

    return parser.parse_args()

def configurar_logging(verbose: bool):
    """Establece el nivel de verbosidad del sistema."""
    nivel = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=nivel,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)

def main():
    args = configurar_cli()
    logger = configurar_logging(args.verbose)

    logger.info(f"Iniciando Pipeline: {args.titulo}")

    try:
        # FASE 1: Preprocesamiento
        logger.info("[1/3] Cargando datos y ejecutando limpieza NLP...")
        df = preprocesamiento.cargar_datos(args.input, args.columna)
        df_limpio = preprocesamiento.procesar_nlp(df, args.columna, args.idioma)

        # FASE 2: Modelado Matemático (Aquí se inyectará el caché de embeddings)
        logger.info("[2/3] Ejecutando modelado semántico y clustering...")
        df_normal, df_outliers = modelado.detectar_outliers_y_ngramas(df_limpio, args.columna)
        df_clasificado = modelado.clasificar_sentimientos(df_normal, args.columna)
        resultados_topicos = modelado.modelar_topicos(df_clasificado)
        resultados_similitud = modelado.analizar_similitud_precio(df_clasificado, args.columna)

        # FASE 3: Generación de Reportes
        logger.info("[3/3] Renderizando visualizaciones interactivas...")
        visualizacion.generar_reporte(
            df_clasificado, 
            df_outliers,
            resultados_topicos, 
            resultados_similitud, 
            args.titulo, 
            args.paleta
        )

        logger.info("Pipeline finalizado exitosamente. Reporte generado.")

    except Exception as e:
        logger.critical(f"Falla catastrófica en el pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()