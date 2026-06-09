# Pipeline NLP Turístico: Análisis de Reseñas y Topología Semántica (Alta Eficiencia)

Este proyecto implementa una arquitectura avanzada de procesamiento de lenguaje natural (NLP) y Machine Learning espacial diseñada para la auditoría de experiencias turísticas a partir de reseñas masivas de usuarios.

El sistema ingesta datasets (optimizados en formato Parquet o CSV), aplica sanitización de datos con control estricto de memoria RAM, extrae polaridad de sentimientos mediante redes neuronales Transformer (`DistilBERT`), y reduce la dimensionalidad de tensores matemáticos para proyectar agrupaciones semánticas (`HDBSCAN` + `UMAP`) en Dashboards HTML interactivos, seguros y accesibles.

## Arquitectura del Sistema

El pipeline ha sido refactorizado a un estándar de grado de producción, operando en tres capas principales diseñadas para entornos con hardware restringido (máquinas virtuales):

### 1. Preprocesamiento y Control de Memoria (`src/preprocesamiento.py`)

* **Gestión Activa de RAM:** Uso explícito de recolección de basura (`gc.collect()`) para destruir tensores y dataframes intermedios, evitando la paginación del sistema operativo.
* **Muestreo Estadístico Representativo:** Implementación de un límite de seguridad (6,000 registros aleatorios) para prevenir el colapso del renderizado web y estabilizar el cálculo espacial matricial.
* **Bifurcación Léxica:** Creación de un corpus con *Stemming* para el cálculo eficiente de distancias euclidianas/coseno, manteniendo simultáneamente una columna intacta (`comentario_grafo`) para garantizar la legibilidad humana en la extracción de N-gramas.

### 2. Modelado Matemático (`src/modelado.py`)

* **Vectorización y Reducción Topológica:** Uso de `SentenceTransformers` (MiniLM) para la generación de embeddings densos. Compresión a un plano bidimensional mediante `UMAP` operando en mono-núcleo (`n_jobs=1`) para erradicar bloqueos mutuos (*deadlocks*) en la CPU.
* **Clustering por Densidad Tolerante:** Aplicación de `HDBSCAN` directamente sobre el plano 2D para evadir la "Maldición de la Dimensionalidad" en datasets pequeños, con mecanismos anticolapso para retener datos ante dispersión extrema.
* **Clasificador de Sentimientos Acelerado:** Motor `DistilBERT` optimizado mediante *Batching* (lotes de 16), deduplicación algorítmica de cadenas repetidas y truncamiento dinámico (`max_length=512`) para tolerancia a fallos por secuencias atípicas.
* **Análisis de Costo-Valor:** Cálculo de Similitud del Coseno entre los embeddings vectoriales de los comentarios y el concepto específico de "precio/costo/valor".

### 3. Inteligencia de Negocios (`src/visualizacion.py`)

* **Renderizado Aislado Seguro:** Generación de componentes Plotly utilizando inyección de dependencias externas (`include_plotlyjs='cdn'`) para prevenir condiciones de carrera (Race Conditions) al abrir archivos HTML locales.
* **Estándares de Accesibilidad:** Abandono del espectro problemático rojo-verde a favor de paletas categóricas de alto contraste lumínico (Azul y Naranja para polaridad), combinadas con mapas de color continuos unificados (`viridis`, `cividis`, `plasma`, `inferno`).
* **Despliegue Multi-Destino:** Generación automatizada de reportes HTML individuales por destino turístico (Huatulco, La Paz, Puerto Vallarta, Riviera Maya, Riviera Nayarit) sin sobreescritura, integrando tablas de componentes reactivos.

---

## Requisitos y Configuración

El proyecto requiere **Python 3.10+**. Para replicar el entorno de ejecución, instale las dependencias contenidas en `requirements.txt`:

```bash
pip install -r requirements.txt

```

*Nota: Durante la primera ejecución, el sistema descargará automáticamente los pesos de los modelos de Hugging Face y los corpus requeridos por NLTK.*

## Ejecución del Pipeline

Los datasets deben colocarse en la carpeta `data/`. El sistema ofrece dos modalidades de ejecución según los requisitos del proyecto:

**Opción A: Ejecución Individual (Cumplimiento de los 5 parámetros)**
Permite ejecutar el orquestador principal de manera quirúrgica, inyectando los parámetros requeridos desde la terminal:

```bash
python src/main.py data/huatulco.csv Comentarios es "Reporte Huatulco" viridis

```

**Opción B: Procesamiento por Lotes**
Utiliza el script disparador para mapear y procesar iterativamente los 5 destinos turísticos, liberando memoria entre cada ejecución:

```bash
python ejecutar_lote.py

```

*Los tableros ejecutivos interactivos se generarán de forma autónoma en el directorio `data/`.*

---

## Consideraciones Teóricas y Limitaciones Conocidas

Para efectos de auditoría técnica académica, se declaran las siguientes decisiones de ingeniería:

1. **Hardware vs Volumen (Muestreo a 6,000 registros):** Procesar más de 40,000 filas con Transformers locales y clustering espacial genera rendimientos decrecientes y un producto inoperable para el navegador web. El límite implementado asegura la validez estadística manteniendo la fluidez a 60 FPS en el cliente final.
2. **Prevención de Deadlocks en CPU:** El forzado de `UMAP` y `HDBSCAN` a operar en un solo hilo computacional (`n_jobs=1`) fue una decisión deliberada frente a la paralelización. Esto incrementa ligeramente el tiempo por iteración, pero garantiza el 100% de estabilidad y completitud en entornos virtualizados o con recursos restringidos (evitando congelamientos permanentes).
3. **Evaluación Espacial en 2D:** Alimentar a HDBSCAN con el plano previamente reducido por UMAP en lugar de la matriz tensorial original de 384 dimensiones resolvió el colapso del clasificador al procesar los archivos individuales de menor volumen.

---

## Créditos y Equipo de Desarrollo

Arquitectura y refactorización del pipeline desarrolladas por:

* **Juan Fernando Santillan Rivera** (`santillanfernando491@gmail.com`)
* **Isaac Pérez Pérez** (`isaacpp6954@gmail.com`)
* **Luis Arturo Hernández Guevara** (`luis.hdez.gue.05@gmail.com`)
