# src/main.py
import argparse
import sys

# Importación de los módulos del equipo
import preprocesamiento
import modelado
import visualizacion

def main():
    # 1. Configuración de parámetros por CLI
    parser = argparse.ArgumentParser(description="Analizador de Comentarios Turísticos")
    parser.add_argument("ruta_csv", type=str, help="Ruta del archivo CSV")
    parser.add_argument("columna_texto", type=str, help="Columna que contiene los comentarios")
    parser.add_argument("idioma", type=str, choices=["es", "en", "fr"], help="Idioma objetivo")
    parser.add_argument("titulo_reporte", type=str, help="Título del reporte de salida")
    parser.add_argument("paleta_colores", type=str, choices=["viridis", "cividis", "plasma", "inferno"], help="Paleta de colores accesible")

    args = parser.parse_args()

    print(f"--- Iniciando Pipeline: {args.titulo_reporte} ---")

    try:
        # FASE 1: Preprocesamiento (Isaac)
        print("[1/3] Ejecutando limpieza y preprocesamiento...")
        df = preprocesamiento.cargar_datos(args.ruta_csv, args.columna_texto)
        df_limpio = preprocesamiento.procesar_nlp(df, args.columna_texto, args.idioma)

        # FASE 2: Modelado Matemático y NLP (Juan Fernando)
        print("[2/3] Ejecutando modelado y análisis semántico...")
        df_normal, df_outliers = modelado.detectar_outliers_y_ngramas(df_limpio, args.columna_texto)
        df_clasificado = modelado.clasificar_sentimientos(df_normal, args.columna_texto)
        resultados_topicos = modelado.modelar_topicos(df_clasificado)
        resultados_similitud = modelado.analizar_similitud_precio(df_clasificado, args.columna_texto)

        # FASE 3: Generación de Reportes y Gráficos (Edgar)
        print("[3/3] Generando visualizaciones interactivas...")
        visualizacion.generar_reporte(
            df_clasificado, 
            df_outliers,
            resultados_topicos, 
            resultados_similitud, 
            args.titulo_reporte, 
            args.paleta_colores
        )

        print("--- Pipeline finalizado exitosamente ---")

    except Exception as e:
        print(f"Error crítico durante la ejecución: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()