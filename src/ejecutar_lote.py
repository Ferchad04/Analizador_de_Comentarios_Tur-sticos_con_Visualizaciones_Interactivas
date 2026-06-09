import subprocess
import sys
import os

# Mapeo exacto basado en los encabezados de tus archivos CSV
configuraciones = [
    ("data/huatulco.csv", "Comentarios", "es", "Reporte_Huatulco", "viridis"),
    ("data/la_paz.csv", "texto", "es", "Reporte_La_Paz", "plasma"),
    ("data/puerto_vallarta.csv", "review_text", "es", "Reporte_Puerto_Vallarta", "inferno"),
    ("data/riviera_maya.csv", "Comentario", "es", "Reporte_Riviera_Maya", "cividis"),
    ("data/riviera_nayarit.csv", "Comentario", "es", "Reporte_Riviera_Nayarit", "viridis")
]

def procesar_lote():
    print("=== Iniciando Procesamiento Individual por Lotes ===")
    
    for ruta, columna, idioma, titulo, paleta in configuraciones:
        if not os.path.exists(ruta):
            print(f"[OMITIDO] No se encontró el archivo: {ruta}")
            continue
            
        print(f"\n>>> Ejecutando pipeline para: {titulo} ({ruta})")
        
        # Llama a main.py respetando los 5 parámetros posicionales
        comando = [sys.executable, "src/main.py", ruta, columna, idioma, titulo, paleta]
        resultado = subprocess.run(comando)
        
        if resultado.returncode != 0:
            print(f"[ERROR] Fallo en el procesamiento de {titulo}.")
            sys.exit(1)

    print("\n=== Procesamiento de todos los archivos finalizado exitosamente ===")

if __name__ == "__main__":
    procesar_lote()