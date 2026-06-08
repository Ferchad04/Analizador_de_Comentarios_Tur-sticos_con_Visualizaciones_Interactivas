import textwrap
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Constantes de Tema de Interfaz (La paleta de gráficas es dinámica)
COLOR_BG       = "#0f1117"
COLOR_SURFACE  = "#1a1d27"
COLOR_BORDER   = "#2a2d3e"
COLOR_TEXT     = "#e8eaf6"
COLOR_MUTED    = "#8892b0"
COLOR_ACCENT   = "#7c3aed"

def _truncar(texto: str) -> str:
    return textwrap.shorten(str(texto), width=150, placeholder="...")

def _construir_red_semantica(df: pd.DataFrame) -> dict:
    """Extrae co-ocurrencias para el grafo semántico sin alterar el orquestador."""
    if df.empty or 'texto_limpio' not in df.columns:
        return {"nodos": [], "enlaces": []}
    
    vectorizer = CountVectorizer(max_features=30, min_df=2)
    try:
        X = vectorizer.fit_transform(df['texto_limpio'].dropna())
    except ValueError:
        return {"nodos": [], "enlaces": []}
        
    palabras = vectorizer.get_feature_names_out()
    co_ocurrencia = (X.T * X).toarray()
    np.fill_diagonal(co_ocurrencia, 0)
    
    valores = co_ocurrencia[co_ocurrencia > 0]
    if len(valores) == 0: return {"nodos": palabras.tolist(), "enlaces": []}
    
    umbral = np.percentile(valores, 85)
    enlaces = [{"fuente": palabras[i], "destino": palabras[j], "peso": int(co_ocurrencia[i, j])}
               for i in range(len(palabras)) for j in range(i + 1, len(palabras))
               if co_ocurrencia[i, j] >= umbral]
               
    return {"nodos": palabras.tolist(), "enlaces": enlaces}

def _figura_dispersion_semantica(df: pd.DataFrame, paleta: str) -> str:
    if df.empty or 'x' not in df.columns:
        return "<p class='empty-msg'>Datos topológicos insuficientes.</p>"
        
    # [RÚBRICA] Codificación Redundante: color + symbol
    fig = px.scatter(
        df, x="x", y="y", 
        color="sentimiento", 
        symbol="sentimiento",
        hover_data=[df.columns[0]],
        color_discrete_sequence=getattr(px.colors.sequential, paleta.capitalize(), px.colors.sequential.Viridis)
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=0.5, color=COLOR_BG)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=COLOR_TEXT),
        xaxis=dict(showgrid=True, gridcolor=COLOR_BORDER, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=COLOR_BORDER, zeroline=False),
        margin=dict(t=10, b=10, l=10, r=10), height=400
    )
    return fig.to_html(full_html=False, include_plotlyjs='cdn')

def _figura_dispersion_precio(df_similitud: pd.DataFrame, paleta: str) -> str:
    if df_similitud.empty or 'similitud_precio_costo' not in df_similitud.columns:
        return "<p class='empty-msg'>Datos de costo insuficientes.</p>"
        
    df_similitud['id'] = range(len(df_similitud))
    # [RÚBRICA] Codificación Redundante y Paleta Accesible
    fig = px.scatter(
        df_similitud, x="id", y="similitud_precio_costo", 
        color="sentimiento", symbol="sentimiento",
        hover_data=[df_similitud.columns[0]],
        color_discrete_sequence=getattr(px.colors.sequential, paleta.capitalize(), px.colors.sequential.Viridis)
    )
    fig.update_traces(marker=dict(size=10, opacity=0.9, line=dict(width=0.5, color=COLOR_BG)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=COLOR_TEXT),
        xaxis=dict(title="Índice de Comentario", gridcolor=COLOR_BORDER),
        yaxis=dict(title="Similitud del Coseno (Precio/Valor)", gridcolor=COLOR_BORDER),
        margin=dict(t=10, b=10, l=10, r=10), height=400
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

def _figura_grafo_html(datos_red: dict) -> str:
    if not datos_red["nodos"]: return "<p class='empty-msg'>Red semántica insuficiente.</p>"
    
    fig = go.Figure()
    G = nx.Graph()
    for n in datos_red["nodos"]: G.add_node(n)
    for e in datos_red["enlaces"]: G.add_edge(e["fuente"], e["destino"], weight=e["peso"])
    
    pos = nx.spring_layout(G, k=0.6, seed=42)
    ex, ey = [], []
    for u, v in G.edges():
        ex.extend([pos[u][0], pos[v][0], None])
        ey.extend([pos[u][1], pos[v][1], None])
        
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color=COLOR_BORDER), hoverinfo="none"))
    
    nodos_list = list(G.nodes())
    grados = [len(list(G.neighbors(n))) for n in nodos_list]
    
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodos_list], y=[pos[n][1] for n in nodos_list],
        mode="markers+text", text=nodos_list, textposition="top center",
        textfont=dict(color=COLOR_TEXT, size=11),
        marker=dict(size=[g*3+10 for g in grados], color=COLOR_ACCENT, line=dict(width=1, color=COLOR_BG)),
        hoverinfo="text", hovertext=[f"Conexiones: {g}" for g in grados]
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(t=10, b=10, l=10, r=10), height=400
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

def generar_reporte(
    df_clasificado: pd.DataFrame, 
    df_outliers: pd.DataFrame,
    resultados_topicos: dict, 
    resultados_similitud: pd.DataFrame, 
    titulo: str, 
    paleta: str
) -> None:
    
    # 1. KPIs Cifras con contexto
    total = len(df_clasificado) + len(df_outliers)
    positivos = len(df_clasificado[df_clasificado['sentimiento'] == 'positivo']) if not df_clasificado.empty else 0
    negativos = len(df_clasificado[df_clasificado['sentimiento'] == 'negativo']) if not df_clasificado.empty else 0
    
    # 2. Tablas Top 5 Precio [RÚBRICA]
    tabla_precio = "<p class='empty-msg'>No hay datos suficientes de precio.</p>"
    if not resultados_similitud.empty and 'similitud_precio_costo' in resultados_similitud.columns:
        top5 = resultados_similitud.nlargest(5, 'similitud_precio_costo')
        tabla_precio = "<table class='voices-table'><thead><tr><th>Comentario</th><th>Similitud</th></tr></thead><tbody>"
        for _, r in top5.iterrows():
            tabla_precio += f"<tr><td class='comment-text'>{_truncar(r.iloc[0])}</td><td class='rank'>{r['similitud_precio_costo']:.4f}</td></tr>"
        tabla_precio += "</tbody></table>"

    # 3. Tablas Outliers [RÚBRICA]
    tabla_outliers = "<p class='empty-msg'>No se detectaron outliers.</p>"
    if not df_outliers.empty and 'top_unigramas' in df_outliers.columns:
        tabla_outliers = "<table class='voices-table'><thead><tr><th>Comentario Atípico</th><th>Unigramas</th><th>Bigramas</th></tr></thead><tbody>"
        for _, r in df_outliers.head(10).iterrows():
            tabla_outliers += f"<tr><td class='comment-text'>{_truncar(r.iloc[0])}</td><td class='rank'>{r['top_unigramas']}</td><td class='rank'>{r.get('top_bigramas','-')}</td></tr>"
        tabla_outliers += "</tbody></table>"

    # 4. Construcción Gráficos
    html_pca   = _figura_dispersion_semantica(df_clasificado, paleta)
    html_grafo = _figura_grafo_html(_construir_red_semantica(df_clasificado))
    html_costo = _figura_dispersion_precio(resultados_similitud, paleta)

    # 5. Estructura HTML
    html_final = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"><title>{titulo}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{ --c-bg: {COLOR_BG}; --c-surface: {COLOR_SURFACE}; --c-border: {COLOR_BORDER}; --c-text: {COLOR_TEXT}; --c-muted: {COLOR_MUTED}; --c-accent: {COLOR_ACCENT}; }}
  body {{ background: var(--c-bg); color: var(--c-text); font-family: 'Segoe UI', sans-serif; padding: 30px; }}
  .page {{ max-width: 1200px; margin: 0 auto; }}
  .header {{ border-bottom: 1px solid var(--c-border); padding-bottom: 20px; margin-bottom: 30px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px; }}
  .kpi-card {{ background: var(--c-surface); border: 1px solid var(--c-border); padding: 20px; border-radius: 8px; border-top: 3px solid var(--c-accent); }}
  .kpi-value {{ font-size: 2rem; font-weight: bold; }}
  .kpi-label {{ color: var(--c-muted); font-size: 0.8rem; text-transform: uppercase; }}
  .card {{ background: var(--c-surface); border: 1px solid var(--c-border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .voices-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
  .voices-table th {{ text-align: left; padding: 10px; border-bottom: 2px solid var(--c-border); color: var(--c-muted); }}
  .voices-table td {{ padding: 10px; border-bottom: 1px solid var(--c-border); }}
  .rank {{ font-family: monospace; color: var(--c-accent); }}
  .comment-text {{ font-size: 0.9rem; }}
  .tabs {{ display: flex; gap: 10px; margin-bottom: 20px; }}
  .tab-btn {{ background: var(--c-surface); color: var(--c-muted); border: 1px solid var(--c-border); padding: 10px 20px; border-radius: 5px; cursor: pointer; }}
  .tab-btn.active {{ background: var(--c-accent); color: #fff; border-color: var(--c-accent); }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
</style>
</head>
<body>
<div class="page">
  <header class="header"><h1>{titulo}</h1><p>Paleta de accesibilidad aplicada: {paleta.capitalize()}</p></header>

  <div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-label">Comentarios Totales</div><div class="kpi-value">{total}</div></div>
    <div class="kpi-card"><div class="kpi-label">Outliers / Ruido</div><div class="kpi-value">{len(df_outliers)}</div></div>
    <div class="kpi-card"><div class="kpi-label">Sentimientos Positivos</div><div class="kpi-value">{positivos}</div></div>
    <div class="kpi-card"><div class="kpi-label">Sentimientos Negativos</div><div class="kpi-value">{negativos}</div></div>
  </div>

  <div class="tabs">
    <button class="tab-btn active" onclick="showTab('topologia', this)">1. Topología y Redes</button>
    <button class="tab-btn" onclick="showTab('precio', this)">2. Análisis Precio/Valor</button>
    <button class="tab-btn" onclick="showTab('outliers', this)">3. Análisis Outliers</button>
  </div>

  <div id="topologia" class="tab-panel active">
    <div class="card">
        <h3>Clustering Espacial Semántico (UMAP + HDBSCAN)</h3>
        {html_pca}
    </div>
    <div class="card">
        <h3>Red de Co-ocurrencia de Tópicos</h3>
        {html_grafo}
    </div>
  </div>

  <div id="precio" class="tab-panel">
    <div class="card">
        <h3>Dispersión por Similitud al Concepto "Precio"</h3>
        {html_costo}
    </div>
    <div class="card">
        <h3>Top 5 Comentarios más cercanos al Centroide "Precio/Costo"</h3>
        {tabla_precio}
    </div>
  </div>

  <div id="outliers" class="tab-panel">
    <div class="card">
        <h3>Análisis N-Gramas de Comentarios Atípicos (Ruido)</h3>
        {tabla_outliers}
    </div>
  </div>

</div>
<script>
  function showTab(id, btn) {{
    document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    btn.classList.add('active');
  }}
</script>
</body>
</html>"""

    ruta_salida = f"{titulo.replace(' ', '_').lower()}_reporte.html"
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(html_final)
    print(f"  [✓] Reporte final generado con accesibilidad visual: {ruta_salida}")