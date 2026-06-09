import os
import re
import pandas as pd
import plotly.express as px

def _crear_tabla_topicos_html(resultados_topicos: dict) -> str:
    """Genera una estructura de tabla HTML limpia para los tópicos detectados."""
    html = """
    <div style='margin-top: 20px; font-family: Arial, sans-serif;'>
        <table style='width:100%; border-collapse: collapse; background-color: #1e1e1e; color: #ffffff;'>
            <thead>
                <tr style='background-color: #2d2d2d; border-bottom: 2px solid #444;'>
                    <th style='padding: 12px; text-align: left;'>Sentimiento</th>
                    <th style='padding: 12px; text-align: left;'>ID Tópico / Método</th>
                    <th style='padding: 12px; text-align: left;'>Palabras Clave</th>
                    <th style='padding: 12px; text-align: left;'>Comentario Representativo</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for sentimiento, datos in resultados_topicos.items():
        if not datos:
            continue
        if datos.get('metodo') == 'frecuencia':
            palabras = ", ".join(datos.get('palabras_clave', []))
            comentario = datos.get('comentario_representativo', 'N/A')
            html += f"""
            <tr style='border-bottom: 1px solid #333;'>
                <td style='padding: 12px; font-weight: bold; color: #ff7f0e;'>{sentimiento.upper()}</td>
                <td style='padding: 12px;'>Frecuencia (Muestra baja)</td>
                <td style='padding: 12px; font-style: italic;'>{palabras}</td>
                <td style='padding: 12px; font-size: 0.9em;'>"{comentario}"</td>
            </tr>
            """
        elif 'topicos' in datos:
            for t_id, t_info in datos['topicos'].items():
                palabras = ", ".join(t_info.get('palabras_clave', []))
                comentario = t_info.get('comentario_representativo', 'N/A')
                html += f"""
                <tr style='border-bottom: 1px solid #333;'>
                    <td style='padding: 12px; font-weight: bold; color: #1f77b4;'>{sentimiento.upper()}</td>
                    <td style='padding: 12px;'>{t_id.upper()}</td>
                    <td style='padding: 12px; font-style: italic;'>{palabras}</td>
                    <td style='padding: 12px; font-size: 0.9em;'>"{comentario}"</td>
                </tr>
                """
    html += "</tbody></table></div>"
    return html

def _crear_tabla_ruido_html(df_outliers: pd.DataFrame) -> str:
    """Genera una tabla HTML con el desglose de n-gramas legibles extraídos del ruido."""
    if df_outliers.empty:
        return "<p style='color: #aaa; font-style: italic;'>No se detectaron anomalías o ruido significativo en este lote.</p>"
        
    html = """
    <div style='margin-top: 20px; font-family: Arial, sans-serif;'>
        <table style='width:100%; border-collapse: collapse; background-color: #1e1e1e; color: #ffffff;'>
            <thead>
                <tr style='background-color: #2d2d2d; border-bottom: 2px solid #444;'>
                    <th style='padding: 12px; text-align: left;'>Muestra de Comentario Atípico</th>
                    <th style='padding: 12px; text-align: left;'>Unigramas Populares</th>
                    <th style='padding: 12px; text-align: left;'>Bigramas Populares</th>
                </tr>
            </thead>
            <tbody>
    """
    
    df_muestra = df_outliers.head(8)
    for _, fila in df_muestra.iterrows():
        comentario_corto = str(fila.iloc[0])[:90] + "..." if len(str(fila.iloc[0])) > 90 else str(fila.iloc[0])
        unigramas = fila.get('top_unigramas', 'N/A')
        bigramas = fila.get('top_bigramas', 'N/A')
        
        html += f"""
        <tr style='border-bottom: 1px solid #333;'>
            <td style='padding: 12px; font-size: 0.9em;'>"{comentario_corto}"</td>
            <td style='padding: 12px; color: #a1c4fd;'>{unigramas}</td>
            <td style='padding: 12px; color: #b7f8db;'>{bigramas}</td>
        </tr>
        """
    html += "</tbody></table></div>"
    return html

def generar_reporte(df_clasificado: pd.DataFrame, df_outliers: pd.DataFrame, resultados_topicos: dict, resultados_similitud: pd.DataFrame, titulo_reporte: str, paleta_colores: str):
    """
    Orquesta la creación de componentes visuales interactivos seguros contra fallos de inicialización.
    """
    print(f"  [Visualización] Construyendo componentes gráficos con la paleta: {paleta_colores}...")

    # Identificar de forma segura la columna con el texto original del usuario
    columnas_tecnicas = {'x', 'y', 'texto_limpio', 'comentario_grafo', 'cluster_hdbscan', 'sentimiento', 'embedding', 'similitud_precio_costo', 'etiqueta_corta'}
    columnas_originales = [c for c in df_clasificado.columns if c not in columnas_tecnicas]
    columna_texto_original = columnas_originales[0] if columnas_originales else df_clasificado.columns[0]

    # 1. Gráfico de Dispersión Espacial (UMAP + HDBSCAN) - Paleta de Alto Contraste
    fig_mapa = px.scatter(
        df_clasificado,
        x='x',
        y='y',
        color='sentimiento',
        hover_data=[columna_texto_original],
        color_discrete_map={'positivo': '#1f77b4', 'negativo': '#ff7f0e'},
        title="Mapeo Semántico Dimensional (Agrupación por Densidad)"
    )
    fig_mapa.update_traces(marker=dict(size=7, opacity=0.85))
    fig_mapa.update_layout(template="plotly_dark", height=500)

    # 2. Gráfico de Distribución de Sentimientos - Paleta de Alto Contraste
    conteo_sentimientos = df_clasificado['sentimiento'].value_counts().reset_index()
    conteo_sentimientos.columns = ['sentimiento', 'total']
    
    fig_sentimiento = px.bar(
        conteo_sentimientos,
        x='sentimiento',
        y='total',
        color='sentimiento',
        color_discrete_map={'positivo': '#1f77b4', 'negativo': '#ff7f0e'},
        title="Distribución General de la Polaridad"
    )
    fig_sentimiento.update_layout(template="plotly_dark", height=350, showlegend=False)

    # 3. Gráfico de Análisis de Costo / Precio
    df_precio_top = resultados_similitud.sort_values(by='similitud_precio_costo', ascending=False).head(10)
    df_precio_top['etiqueta_corta'] = df_precio_top[columna_texto_original].astype(str).str.slice(0, 40) + "..."
    
    fig_precio = px.bar(
        df_precio_top,
        x='similitud_precio_costo',
        y='etiqueta_corta',
        orientation='h',
        title="Top 10 Comentarios Críticos Asociados a Costo/Precio/Valor",
        color='similitud_precio_costo',
        color_continuous_scale=paleta_colores
    )
    fig_precio.update_layout(template="plotly_dark", height=350, yaxis={'categoryorder': 'total ascending'})

    # Compilación HTML con renderizadores aislados seguros
    tabla_topicos_html = _crear_tabla_topicos_html(resultados_topicos)
    tabla_ruido_html = _crear_tabla_ruido_html(df_outliers)

    nombre_archivo = re.sub(r'[^\w\s]', '', titulo_reporte).replace(' ', '_').lower()
    ruta_salida = f"data/{nombre_archivo}.html"

    # CRÍTICO: include_plotlyjs='cdn' inserta scripts autogestionados independientes
    html_layout = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>{titulo_reporte}</title>
        <style>
            body {{ background-color: #121212; color: #ffffff; font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; padding: 20px; background-color: #1e1e1e; border-bottom: 3px solid #333; margin-bottom: 20px; }}
            .authors {{ color: #aaa; font-size: 0.9em; margin-top: 10px; }}
            .section {{ background-color: #1a1a1a; padding: 20px; margin-bottom: 20px; border-radius: 5px; border: 1px solid #2d2d2d; }}
            .row {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .col-6 {{ flex: 1; min-width: 45%; }}
            h2 {{ color: #ffffff; border-left: 5px solid #4a90e2; padding-left: 10px; margin-top: 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{titulo_reporte}</h1>
                <p>Pipeline Analítico de Sentimientos y Modelado de Tópicos Espaciales</p>
                <div class="authors">
                    <strong>Desarrollado por:</strong><br>
                    Juan Fernando Santillan Rivera (santillanfernando491@gmail.com) &bull; 
                    Isaac Pérez Pérez (isaacpp6954@gmail.com) &bull; 
                    Luis Arturo Hernández Guevara (luis.hdez.gue.05@gmail.com)
                </div>
            </div>

            <div class="section">
                <h2>1. Cartografía Semántica del Destino</h2>
                {fig_mapa.to_html(full_html=False, include_plotlyjs='cdn')}
            </div>

            <div class="section">
                <div class="row">
                    <div class="col-6">
                        <h2>2. Análisis de Sentimientos</h2>
                        {fig_sentimiento.to_html(full_html=False, include_plotlyjs='cdn')}
                    </div>
                    <div class="col-6">
                        <h2>3. Métrica de Relación Costo-Precio</h2>
                        {fig_precio.to_html(full_html=False, include_plotlyjs='cdn')}
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>4. Estructura Temática (Modelado de Tópicos)</h2>
                {tabla_topicos_html}
            </div>

            <div class="section">
                <h2>5. Análisis de Comentarios Atípicos (Detección de Ruido)</h2>
                {tabla_ruido_html}
            </div>
        </div>
    </body>
    </html>
    """

    os.makedirs("data", exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(html_layout)

    print(f"  [OK] Dashboard interactivo exportado exitosamente en: {ruta_salida}")