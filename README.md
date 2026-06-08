# Pipeline NLP Turístico: Análisis de Reseñas y Topología Semántica

Este proyecto implementa una arquitectura de procesamiento de lenguaje natural (NLP) diseñada para la auditoría de experiencias turísticas a partir de reseñas de usuarios. 

El sistema ingesta datasets en formato CSV, aplica limpieza de datos, extrae polaridad de sentimientos mediante un clasificador léxico ponderado y reduce la dimensionalidad de tensores para proyectar agrupaciones semánticas en un Dashboard HTML interactivo y 100% offline.

## Arquitectura del Sistema

El pipeline se divide en tres capas principales operando de manera secuencial:

### 1. Preprocesamiento (`src/preprocesamiento.py`)
- **Deduplicación:** Eliminación estricta de registros duplicados idénticos para evitar sesgos estadísticos en el modelado.
- **Bifurcación de Tokens:** Generación de un corpus truncado (`comentario_nlp`, con *stemming* y sin tildes) para eficiencia matemática en HDBSCAN y PCA, y preservación del texto original limpio (`comentario_grafo`) para garantizar la legibilidad humana en la topología de red.

### 2. Modelado Matemático (`src/modelado.py`)
- **Agrupamiento Semántico:** Uso de `SentenceTransformers` (forzado a CPU) y `HDBSCAN` para detección de ruido estructural y anomalías espaciales.
- **Clasificador de Sentimientos:** Sistema determinista basado en reglas (Analizador Léxico Ponderado). Utiliza diccionarios de alta densidad organizados en niveles de intensidad (ej. *Fuerte* vs *Normal*) combinados con ventanas de retroceso (*lookback windows*) para identificar inversiones de polaridad (ej. negaciones antes de adjetivos).
- **Topología de Red:** Generación de matriz de co-ocurrencia extraída mediante `CountVectorizer` y estructurada con `NetworkX`. Implementa un umbral dinámico basado en percentiles para evitar la saturación visual en *datasets* masivos.
- **Caché Computacional:** Integración de `joblib` para persistir tensores pesados en disco (`cache_nlp/`) y minimizar drásticamente la latencia en re-ejecuciones.

### 3. Inteligencia de Negocios (`src/visualizacion.py`)
- **Renderizado Dinámico:** Generación de componentes interactivos (Dona, Scatter 2D, Grafo) utilizando `Plotly`.
- **Despliegue Autónomo:** Inyección directa del bundle de JavaScript en el documento, garantizando que el entregable final sea un HTML estático, independiente y libre de dependencias a CDNs externos.

## Requisitos y Configuración

El proyecto fue desarrollado y validado en Python 3.10+. Para replicar el entorno de ejecución, instale las dependencias contenidas en `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Nota inicial: Al ejecutarse por primera vez, el sistema descargará automáticamente el modelo multilingüe de `sentence-transformers` y los corpus requeridos por `nltk` (`punkt`, `stopwords`).*

## Ejecución del Pipeline

Coloque los datasets en la carpeta `data/` respetando el formato CSV y asegurando la existencia de la columna objetivo de texto crudo (configurada por defecto en el código).

Ejecute el orquestador principal desde la raíz del proyecto:

```bash
python src/main.py
```

Los tableros ejecutivos serán generados en el directorio actual con el prefijo `reporte_ejecutivo_` en formato `.html`.

## Consideraciones Teóricas y Limitaciones Conocidas

Para efectos de auditoría técnica académica, se declaran las siguientes limitaciones de la arquitectura implementada:

* **Ceguera Sintáctica Profunda:** El clasificador de sentimientos emplea una arquitectura *Bag-of-Words* combinada con pesos heurísticos. Aunque es altamente preciso en reseñas cortas gracias a la *lookback window*, es susceptible a dependencias sintácticas largas o sarcasmo complejo que un modelo de *Deep Learning* puro (tipo RoBERTa) sí capturaría. Esta decisión fue pragmática para garantizar rendimiento computacional en CPU y evitar la clasificación errónea por falsos contextos semánticos del Transformer original.
* **Agresividad de HDBSCAN:** Parámetros como `min_cluster_size` fueron ajustados para optimizar la representación espacial. En la versión actual, se neutralizó la eliminación destructiva de `HDBSCAN` en la capa de modelado, preservando el 100% de la limpieza inicial del ETL para evitar la pérdida de representatividad estadística (Hemorragia de Datos) en corpus con vocabularios muy dispersos.
* **Ausencia de Series Temporales:** Debido a la naturaleza estática de los datasets extraídos en la fase de origen, el modelo omite el análisis longitudinal, limitando la capacidad de identificar estacionalidades operativas en la industria turística.