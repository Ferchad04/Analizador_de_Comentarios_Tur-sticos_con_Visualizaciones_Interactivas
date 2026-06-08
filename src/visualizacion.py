# =============================================================================
# src/visualizacion.py
# Generador de Dashboard HTML offline.
#
# CHANGELOG:
#   - Corrección ortográfica estricta (tildes y letra ñ).
#   - Eliminación de métricas, botones y tablas referentes a Precio.
#   - Reactivación de herramientas interactivas de Plotly (zoom, pan, png).
# =============================================================================

import textwrap
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import networkx as nx
from typing import Optional

# Paleta Corporativa y SVGs
COLOR_POSITIVO = "#2ecc71"   
COLOR_NEGATIVO = "#e74c3c"   
COLOR_NEUTRAL  = "#3b5bdb"   
COLOR_BG       = "#0f1117"
COLOR_SURFACE  = "#1a1d27"
COLOR_BORDER   = "#2a2d3e"
COLOR_TEXT     = "#e8eaf6"
COLOR_MUTED    = "#8892b0"
COLOR_ACCENT   = "#7c3aed"

MAPA_COLORES = {
    "positivo": COLOR_POSITIVO,
    "negativo": COLOR_NEGATIVO,
    "neutral":  COLOR_NEUTRAL,
}

def _svg_icon(path_d: str, size: int = 16, color: str = "currentColor") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:middle;margin-right:6px">'
        f'<path d="{path_d}"/></svg>'
    )

_ICON_CHART    = _svg_icon("M18 20V10M12 20V4M6 20v-6", color=COLOR_TEXT)
_ICON_SEARCH   = _svg_icon("M11 19a8 8 0 100-16 8 8 0 000 16zm9 2l-4.35-4.35", color=COLOR_TEXT)
_ICON_ALERT    = _svg_icon("M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z", color=COLOR_NEGATIVO)
_ICON_STAR     = _svg_icon("M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z", color=COLOR_POSITIVO)
_ICON_COMPASS  = _svg_icon("M12 2a10 10 0 100 20 10 10 0 000-20zm4.9 4.9l-4.24 9.19-9.19 4.24 4.24-9.19 9.19-4.24z", size=24, color="#fff")

# Constructores Plotly
def _figura_dona(df: pd.DataFrame) -> Optional[go.Figure]:
    if "sentimiento" not in df.columns or df.empty:
        return None

    conteo  = df["sentimiento"].value_counts().reset_index()
    conteo.columns = ["sentimiento", "cantidad"]
    colores = [MAPA_COLORES.get(s, "#555") for s in conteo["sentimiento"]]

    fig = go.Figure(go.Pie(
        labels=conteo["sentimiento"].str.capitalize(),
        values=conteo["cantidad"],
        hole=0.62,
        marker=dict(colors=colores, line=dict(color=COLOR_BG, width=3)),
        textfont=dict(size=13, color=COLOR_TEXT),
        hovertemplate="<b>%{label}</b><br>%{value} reseñas (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color=COLOR_TEXT, size=12), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=10, l=10, r=10),
        height=320,
        annotations=[dict(
            text=f"<b>{len(df)}</b><br><span style='font-size:11px'>reseñas</span>",
            x=0.5, y=0.5, font_size=18, font_color=COLOR_TEXT, showarrow=False,
        )],
    )
    return fig

def _figura_pca(datos_pca: dict) -> Optional[go.Figure]:
    if not datos_pca.get("componentes") or not datos_pca.get("etiquetas"):
        return None

    componentes = datos_pca["componentes"]
    etiquetas   = datos_pca["etiquetas"]

    df_pca = pd.DataFrame({
        "PC1": [c[0] for c in componentes],
        "PC2": [c[1] if len(c) > 1 else 0 for c in componentes],
        "Sentimiento": [e.capitalize() for e in etiquetas],
    })

    fig = px.scatter(
        df_pca, x="PC1", y="PC2", color="Sentimiento",
        color_discrete_map={k.capitalize(): v for k, v in MAPA_COLORES.items()},
        opacity=0.75,
        hover_data={"PC1": ":.3f", "PC2": ":.3f"},
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color=COLOR_BG)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXT),
        xaxis=dict(showgrid=True, gridcolor=COLOR_BORDER, zeroline=False, title="Componente 1"),
        yaxis=dict(showgrid=True, gridcolor=COLOR_BORDER, zeroline=False, title="Componente 2"),
        legend=dict(title="", font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=40, l=40, r=10),
        height=380,
    )
    return fig

def _figura_grafo(datos_red: dict) -> go.Figure:
    fig = go.Figure()
    if not datos_red or not datos_red.get("nodos"):
        fig.add_annotation(
            text="Datos de red insuficientes para generar la topología",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color=COLOR_MUTED, size=13),
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
        return fig

    G = nx.Graph()
    for nodo in datos_red["nodos"]:
        G.add_node(nodo)
    for enlace in datos_red["enlaces"]:
        G.add_edge(enlace["fuente"], enlace["destino"], weight=enlace["peso"])

    pos = nx.spring_layout(G, k=0.6, seed=42)

    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ex.extend([x0, x1, None])
        ey.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(width=0.8, color=COLOR_BORDER),
        hoverinfo="none",
    ))

    nodos_list = list(G.nodes())
    grados     = [len(list(G.neighbors(n))) for n in nodos_list]
    tamanios   = [g * 2.5 + 10 for g in grados]

    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodos_list],
        y=[pos[n][1] for n in nodos_list],
        mode="markers+text",
        text=nodos_list,
        textposition="top center",
        textfont=dict(size=10, color=COLOR_TEXT),
        hovertext=[f"<b>{n}</b><br>Conexiones: {g}" for n, g in zip(nodos_list, grados)],
        hoverinfo="text",
        marker=dict(
            size=tamanios,
            color=grados,
            colorscale=[[0, COLOR_NEUTRAL], [0.5, COLOR_ACCENT], [1, COLOR_POSITIVO]],
            showscale=False,
            line=dict(width=1.5, color=COLOR_BG),
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(b=10, l=5, r=5, t=10),
        height=380,
    )
    return fig

def _serializar_figuras(fig_dona, fig_pca, fig_grafo) -> tuple[str, str, str]:
    FALLBACK = f"<p style='color:{COLOR_MUTED};text-align:center;padding:40px 0'>Datos insuficientes para esta visualización.</p>"

    # displayModeBar en True para habilitar herramientas interactivas Plotly
    plotly_config = {"displayModeBar": True, "responsive": True}

    if fig_dona is not None:
        html_dona = fig_dona.to_html(full_html=False, include_plotlyjs=True, config=plotly_config)
    else:
        html_dona = FALLBACK

    if fig_pca is not None:
        html_pca = fig_pca.to_html(full_html=False, include_plotlyjs=False, config=plotly_config)
    else:
        html_pca = FALLBACK

    html_grafo = fig_grafo.to_html(full_html=False, include_plotlyjs=False, config=plotly_config)

    return html_dona, html_pca, html_grafo

# Constructores de Tablas
def _top_n_por_categoria(df: pd.DataFrame, columna_original: str, n: int = 4) -> dict:
    resultado = {"negativo": [], "positivo": []}
    if df.empty or columna_original not in df.columns:
        return resultado

    def _truncar(texto: str) -> str:
        return textwrap.shorten(str(texto), width=300, placeholder="…")

    if "sentimiento" in df.columns:
        for cat in ("negativo", "positivo"):
            subset = df[df["sentimiento"] == cat][[columna_original]].dropna()
            resultado[cat] = [_truncar(t) for t in subset[columna_original].head(n).tolist()]

    return resultado

def _construir_tabla_html(filas: list) -> str:
    if not filas:
        return "<p class='empty-msg'>No hay comentarios disponibles en esta categoría.</p>"

    filas_html = ""
    for i, item in enumerate(filas, 1):
        filas_html += f"""
        <tr>
            <td class="rank">#{i}</td>
            <td class="comment-text">{item}</td>
        </tr>"""

    return f"""
    <table class="voices-table">
        <thead><tr><th>#</th><th>Comentario</th></tr></thead>
        <tbody>{filas_html}</tbody>
    </table>"""

# KPIs
def _calcular_kpis(df: pd.DataFrame, df_outliers: pd.DataFrame) -> dict:
    total     = len(df) + len(df_outliers)
    procesado = len(df)
    tasa      = round(procesado / total * 100, 1) if total > 0 else 0.0

    conteo = {"positivo": 0, "negativo": 0, "neutral": 0}
    if "sentimiento" in df.columns:
        vc = df["sentimiento"].value_counts()
        for k in conteo:
            conteo[k] = int(vc.get(k, 0))

    return {
        "total_ingestado" : total,
        "total_procesado" : procesado,
        "ruido_descartado": len(df_outliers),
        "tasa_limpieza"   : tasa,
        "positivos"       : conteo["positivo"],
        "negativos"       : conteo["negativo"],
        "neutrales"       : conteo["neutral"],
    }

# Template HTML Maestro
def _construir_html(
    titulo: str, kpis: dict, html_dona: str, html_pca: str, 
    html_grafo: str, tabla_quejas: str, tabla_positivas: str
) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{titulo}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --c-bg      : {COLOR_BG};
    --c-surface : {COLOR_SURFACE};
    --c-border  : {COLOR_BORDER};
    --c-text    : {COLOR_TEXT};
    --c-muted   : {COLOR_MUTED};
    --c-pos     : {COLOR_POSITIVO};
    --c-neg     : {COLOR_NEGATIVO};
    --c-neu     : {COLOR_NEUTRAL};
    --c-accent  : {COLOR_ACCENT};
    --radius    : 12px;
    --radius-sm : 8px;
    --font      : 'Segoe UI', system-ui, sans-serif;
    --mono      : 'Cascadia Code', 'Fira Code', monospace;
  }}
  html, body {{ background: var(--c-bg); color: var(--c-text); font-family: var(--font); font-size: 15px; line-height: 1.6; min-height: 100vh; }}
  .page {{ max-width: 1300px; margin: 0 auto; padding: 32px 24px 72px; }}
  
  .header {{ margin-bottom: 36px; padding-bottom: 20px; border-bottom: 1px solid var(--c-border); }}
  .header h1 {{ font-size: 1.8rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.2; margin-bottom: 8px; }}
  .header p {{ color: var(--c-muted); font-size: .95rem; }}
  
  .kpi-grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 32px; }}
  @media (max-width: 1000px) {{ .kpi-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
  @media (max-width: 600px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}

  .kpi-card {{ background: var(--c-surface); border: 1px solid var(--c-border); border-radius: var(--radius); padding: 20px 18px 16px; position: relative; overflow: hidden; transition: border-color .2s, transform .2s; animation: fadeUp .4s ease both; }}
  .kpi-card:hover {{ border-color: var(--c-accent); transform: translateY(-2px); }}
  .kpi-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--accent-clr, var(--c-accent)); border-radius: var(--radius) var(--radius) 0 0; }}
  .kpi-label {{ font-size: .72rem; color: var(--c-muted); text-transform: uppercase; letter-spacing: .07em; margin-bottom: 8px; }}
  .kpi-value {{ font-size: 2rem; font-weight: 800; line-height: 1; }}
  .kpi-sub   {{ font-size: .77rem; color: var(--c-muted); margin-top: 6px; }}
  
  .tabs-nav {{ display: flex; gap: 6px; margin-bottom: 28px; background: var(--c-surface); border: 1px solid var(--c-border); border-radius: var(--radius); padding: 6px; width: fit-content; }}
  .tab-btn {{ background: transparent; border: none; cursor: pointer; color: var(--c-muted); font: 600 .88rem var(--font); padding: 9px 22px; border-radius: var(--radius-sm); transition: all .18s; letter-spacing: .02em; display: flex; align-items: center; gap: 8px; }}
  .tab-btn.active {{ background: var(--c-accent); color: #fff; box-shadow: 0 2px 12px rgba(124,58,237,.4); }}
  .tab-btn:hover:not(.active) {{ color: var(--c-text); background: var(--c-border); }}
  .tab-panel {{ display: none; }}
  .tab-panel.active {{ display: block; }}
  
  .card {{ background: var(--c-surface); border: 1px solid var(--c-border); border-radius: var(--radius); padding: 24px; margin-bottom: 20px; animation: fadeUp .45s ease both; }}
  .card-title {{ font-size: 1rem; font-weight: 700; letter-spacing: -.01em; }}
  .card-insight {{ font-size: .82rem; color: var(--c-muted); margin: 10px 0 18px; border-left: 3px solid var(--c-accent); padding-left: 12px; line-height: 1.6; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 820px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  
  .voices-nav {{ display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap; }}
  .voice-btn {{ background: var(--c-border); border: 1px solid transparent; border-radius: 20px; color: var(--c-muted); font: 600 .82rem var(--font); padding: 7px 16px; cursor: pointer; transition: all .18s; letter-spacing: .03em; display: flex; align-items: center; gap: 6px; }}
  .voice-btn:hover {{ color: var(--c-text); border-color: var(--c-accent); }}
  .voice-btn.vb-neg {{ background: rgba(231,76,60,.15);  color: var(--c-neg); border-color: var(--c-neg); }}
  .voice-btn.vb-pos {{ background: rgba(46,204,113,.15); color: var(--c-pos); border-color: var(--c-pos); }}
  .voices-panel {{ display: none; }}
  .voices-panel.active {{ display: block; }}
  
  .voices-table {{ width: 100%; border-collapse: collapse; font-size: .875rem; }}
  .voices-table thead tr {{ border-bottom: 2px solid var(--c-border); }}
  .voices-table th {{ color: var(--c-muted); font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; padding: 0 12px 10px; text-align: left; font-weight: 600; }}
  .voices-table td {{ padding: 13px 12px; border-bottom: 1px solid var(--c-border); vertical-align: top; }}
  .voices-table tbody tr:last-child td {{ border-bottom: none; }}
  .voices-table tbody tr:hover {{ background: rgba(255,255,255,.022); }}
  .rank {{ color: var(--c-muted); font-family: var(--mono); font-size: .8rem; width: 36px; white-space: nowrap; }}
  .comment-text {{ color: var(--c-text); line-height: 1.55; }}
  .empty-msg {{ color: var(--c-muted); font-style: italic; text-align: center; padding: 28px 0; font-size: .88rem; }}
  
  @keyframes fadeUp {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: translateY(0); }} }}
</style>
</head>
<body>
<div class="page">
  <header class="header">
    <h1>{titulo}</h1>
    <p>Pipeline NLP Turístico &middot; Análisis de Reseñas automatizado</p>
  </header>

  <section class="kpi-grid">
    <div class="kpi-card" style="--accent-clr:var(--c-text)">
      <div class="kpi-label">Total Ingestado</div>
      <div class="kpi-value">{kpis['total_ingestado']:,}</div>
      <div class="kpi-sub">reseñas en dataset original</div>
    </div>
    <div class="kpi-card" style="--accent-clr:var(--c-pos)">
      <div class="kpi-label">Procesadas</div>
      <div class="kpi-value" style="color:var(--c-pos)">{kpis['total_procesado']:,}</div>
      <div class="kpi-sub">{kpis['tasa_limpieza']}% tasa de limpieza</div>
    </div>
    <div class="kpi-card" style="--accent-clr:var(--c-neg)">
      <div class="kpi-label">Ruido Descartado</div>
      <div class="kpi-value" style="color:var(--c-neg)">{kpis['ruido_descartado']:,}</div>
      <div class="kpi-sub">outliers, duplicados y vacíos</div>
    </div>
    <div class="kpi-card" style="--accent-clr:var(--c-pos)">
      <div class="kpi-label">Positivas</div>
      <div class="kpi-value" style="color:var(--c-pos)">{kpis['positivos']:,}</div>
      <div class="kpi-sub">reseñas de alta satisfacción</div>
    </div>
    <div class="kpi-card" style="--accent-clr:var(--c-neg)">
      <div class="kpi-label">Negativas</div>
      <div class="kpi-value" style="color:var(--c-neg)">{kpis['negativos']:,}</div>
      <div class="kpi-sub">reseñas críticas o alertas</div>
    </div>
  </section>

  <nav class="tabs-nav" role="tablist">
    <button class="tab-btn active" onclick="switchTab('vista-general',this)" role="tab">
      Vista General
    </button>
    <button class="tab-btn" onclick="switchTab('exploracion',this)" role="tab">
      Exploración de Datos
    </button>  
  </nav>

  <div id="tab-vista-general" class="tab-panel active">
    <div class="two-col">
      <div class="card">
        <div class="card-title">Distribución de Sentimientos</div>
        <p class="card-insight">
          Proporción de reseñas clasificadas por polaridad. Un volumen de negativas
          superior al 30% indica deficiencias sistémicas que requieren atención.
        </p>
        {html_dona}
      </div>
      <div class="card">
        <div class="card-title">Voces Críticas del Viajero</div>
        <p class="card-insight">
          Explora los comentarios representativos. Identifica patrones recurrentes 
          para respaldar decisiones operativas basadas en datos.
        </p>
        <div class="voices-nav">
          <button class="voice-btn vb-neg" onclick="switchVoice('quejas',this)">
            Quejas Principales
          </button>
          <button class="voice-btn" onclick="switchVoice('positivas',this)">
            Positivas
          </button>
        </div>
        <div id="voice-quejas" class="voices-panel active">{tabla_quejas}</div>
        <div id="voice-positivas" class="voices-panel">{tabla_positivas}</div>
      </div>
    </div>
  </div>

  <div id="tab-exploracion" class="tab-panel">
    <div class="card">
      <div class="card-title">Proyección PCA — Mapa de Sentimientos en 2D</div>
      <p class="card-insight">
        <strong>Cómo leer esta gráfica:</strong> Cada punto es una reseña reducida a coordenadas vectoriales (X, Y).
        Agrupaciones densas del mismo color señalan contextos y problemas idénticos reportados por múltiples clientes.
        Si las quejas (rojas) se concentran aisladas del resto, existe una causa raíz unificada. Puntos dispersos son 
        casos aislados sin significancia estadística general.
      </p>
      {html_pca}
    </div>
    <div class="card">
      <div class="card-title">Topología Semántica — Red de Co-ocurrencia de Conceptos</div>
      <p class="card-insight">
        <strong>Cómo leer esta gráfica:</strong> Los nodos representan términos dominantes del corpus. 
        Una arista (línea) indica que los turistas utilizan ambas palabras en la misma oración frecuentemente.
        El diámetro del nodo refleja su centralidad en la conversación global del destino. 
      </p>
      {html_grafo}
    </div>
  </div>
</div>

<script>
  function switchTab(id, btn) {{
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + id).classList.add('active');
    btn.classList.add('active');
  }}

  const VOICE_MAP = {{
    quejas   : 'vb-neg',
    positivas: 'vb-pos',
  }};

  function switchVoice(id, btn) {{
    document.querySelectorAll('.voices-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.voice-btn').forEach(b => b.classList.remove('vb-neg','vb-pos'));
    document.getElementById('voice-' + id).classList.add('active');
    btn.classList.add(VOICE_MAP[id]);
  }}
</script>
</body>
</html>"""

def generar_reporte(
    df_clasificado: pd.DataFrame,
    df_outliers: pd.DataFrame,
    datos_modelo: dict,
    titulo: str,
    columna_original: str,
    ruta_salida: str,
) -> None:
    df_vis = datos_modelo.get("df_procesado", df_clasificado)
    df_outliers_total = pd.concat(
        [df_outliers, datos_modelo.get("df_outliers_semanticos", pd.DataFrame())],
        ignore_index=True,
    )
    
    kpis = _calcular_kpis(df_vis, df_outliers_total)
    
    fig_dona  = _figura_dona(df_vis)
    fig_pca   = _figura_pca(datos_modelo.get("pca", {}))
    fig_grafo = _figura_grafo(datos_modelo.get("datos_red", {}))

    html_dona, html_pca, html_grafo = _serializar_figuras(fig_dona, fig_pca, fig_grafo)

    top_n = _top_n_por_categoria(df_vis, columna_original, n=4)
    tabla_quejas    = _construir_tabla_html(top_n["negativo"])
    tabla_positivas = _construir_tabla_html(top_n["positivo"])

    html_final = _construir_html(
        titulo          = titulo,
        kpis            = kpis,
        html_dona       = html_dona,
        html_pca        = html_pca,
        html_grafo      = html_grafo,
        tabla_quejas    = tabla_quejas,
        tabla_positivas = tabla_positivas,
    )

    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(html_final)
    print(f"  [HTML] Dashboard escrito: {ruta_salida}")