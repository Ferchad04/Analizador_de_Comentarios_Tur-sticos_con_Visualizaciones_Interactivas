# src/visualizacion.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict
import os

def generar_reporte(
    df_clasificado: pd.DataFrame, 
    df_outliers: pd.DataFrame,
    resultados_topicos: Dict[str, dict], 
    resultados_similitud: pd.DataFrame, 
    titulo: str, 
    paleta: str
) -> None:
    """
    Genera un reporte interactivo en formato HTML con gráficos de dispersión independientes
    y análisis de texto offline.
    """
    print(f"\n--- Generando Reporte: {titulo} ---")
    
    # 1. Gráfico de Dispersión por Tópicos (Reducción de dimensionalidad lineal PCA)
    fig_topicos = go.Figure()
    if not df_clasificado.empty and 'sentimiento' in df_clasificado.columns:
        # Generar coordenadas X/Y para el scatter plot usando las relaciones de palabras
        tfidf = TfidfVectorizer(max_features=100)
        vectores = tfidf.fit_transform(df_clasificado[df_clasificado.columns[0]]).toarray()
        
        # Reducir a 2 dimensiones para el gráfico de dispersión
        pca = PCA(n_components=2, random_state=42)
        coordenadas = pca.fit_transform(vectores)
        df_clasificado['x'] = coordenadas[:, 0]
        df_clasificado['y'] = coordenadas[:, 1]
        
        # Codificación redundante: Color por sentimiento y Símbolo por tipo de sentimiento [cite: 42, 367]
        fig_topicos = px.scatter(
            df_clasificado,
            x='x',
            y='y',
            color='sentimiento',
            symbol='sentimiento',
            hover_data=[df_clasificado.columns[0]],
            title="Distribución Semántica de Comentarios por Sentimiento",
            color_discrete_sequence=getattr(px.colors.sequential, paleta.capitalize(), px.colors.sequential.Viridis)
        )
        fig_topicos.update_traces(marker=dict(size=10))

    # 2. Gráfico de Dispersión específico para el concepto "Precio / Valor / Costo" [cite: 44, 46]
    fig_precio = go.Figure()
    if not resultados_similitud.empty and 'similitud_precio_costo' in resultados_similitud.columns:
        resultados_similitud['indice'] = range(len(resultados_similitud))
        
        fig_precio = px.scatter(
            resultados_similitud,
            x='indice',
            y='similitud_precio_costo',
            color='sentimiento',
            hover_data=[resultados_similitud.columns[0]],
            title="Relación de Comentarios con el Concepto 'Precio / Valor / Costo'",
            labels={'similitud_precio_costo': 'Similitud Semántica (Coseno)', 'indice': 'Identificador de Comentario'},
            color_discrete_sequence=getattr(px.colors.sequential, paleta.capitalize(), px.colors.sequential.Viridis)
        )
        fig_precio.update_traces(marker=dict(size=12))

    # 3. Extracción del Top 5 de comentarios sobre Precio/Valor/Costo [cite: 47]
    html_top5 = "<h3>No hay datos suficientes para el análisis de costo.</h3>"
    if 'similitud_precio_costo' in resultados_similitud.columns:
        top5 = resultados_similitud.nlargest(5, 'similitud_precio_costo')
        html_top5 = "<table border='1' style='border-collapse: collapse; width: 100%; text-align: left;'>"
        html_top5 += "<tr style='background-color: #f2f2f2;'><th>Comentario</th><th>Sentimiento</th><th>Similitud</th></tr>"
        for _, row in top5.iterrows():
            html_top5 += f"<tr><td>{row.iloc[0]}</td><td>{row['sentimiento']}</td><td>{row['similitud_precio_costo']:.4f}</td></tr>"
        html_top5 += "</table>"

    # 4. Compilación del Reporte HTML Autónomo (Offline) [cite: 59, 228]
    # include_plotlyjs=True inyecta la librería completa eliminando dependencias de red [cite: 59]
    div_topicos = fig_topicos.to_html(full_html=False, include_plotlyjs='cdn')
    div_precio = fig_precio.to_html(full_html=False, include_plotlyjs=False)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>{titulo}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; background-color: #fafafa; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1, h2, h3 {{ color: #222; }}
            .graph-container {{ margin: 40px 0; }}
            table {{ margin-top: 20px; }}
            th, td {{ padding: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{titulo}</h1>
            <p>Análisis de lenguaje natural ejecutado localmente de forma autónoma.</p>
            
            <div class="graph-container">
                <h2>1. Agrupación por Sentimientos</h2>
                {div_topicos}
            </div>

            <div class="graph-container">
                <h2>2. Análisis Semántico de Precio, Valor y Costo</h2>
                {div_precio}
            </div>

            <div class="graph-container">
                <h2>3. Top 5 Comentarios más Relevantes al Concepto "Precio"</h2>
                {html_top5}
            </div>
        </div>
    </body>
    </html>
    """

    # Guardar el archivo final en la raíz del proyecto o directorio asignado
    nombre_archivo = f"{titulo.replace(' ', '_').lower()}_reporte.html"
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Reporte interactivo guardado exitosamente como: {nombre_archivo}")