# Valuador inmobiliario CABA

Este repositorio contiene un MVP de un valuador inmobiliario para inmuebles en la Ciudad de Buenos Aires. El flujo completo descarga el dataset desde Kaggle, realiza limpieza y exploración con SQL sobre DuckDB, enriquece las variables con PySpark, entrena varios modelos de machine learning y expone una aplicación interactiva en Streamlit.

## Requisitos

1. **Python 3.10+**.
2. Dependencias del proyecto:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Credenciales de Kaggle** configuradas en `~/.kaggle/kaggle.json`. El módulo [`kagglehub`](https://pypi.org/project/kagglehub/) utiliza esas credenciales automáticamente para descargar los datos.

4. Para ejecutar PySpark localmente se necesita Java (OpenJDK 8+). En la mayoría de distribuciones Linux se puede instalar con `sudo apt install default-jdk`.

## Estructura de carpetas

```
├── app_streamlit/          # Aplicación Streamlit
├── data/                   # Archivos generados por DuckDB y Spark
├── models/                 # Modelo entrenado y metadatos
├── reports/                # Métricas y resúmenes
├── samples/                # CSV sintético para pruebas rápidas
└── src/                    # Código del pipeline
```

Los artefactos generados (`data/`, `models/`, `reports/`) están ignorados por Git para evitar subir archivos voluminosos. El pipeline los crea automáticamente.

## Ejecución del pipeline completo

El pipeline se ejecuta desde el entrypoint `src/run_pipeline.py`. Por defecto descargará el dataset de Kaggle, generará `data/props_model.parquet` con DuckDB, procesará con Spark a `data/props_model_sparked.parquet`, entrenará los modelos y guardará los artefactos finales.

```bash
python -m src.run_pipeline
```

Pasos principales:

1. **Descarga** (`kagglehub`): descarga y descomprime los archivos en `data/raw/`. Si el dataset sólo provee archivos `.pkl`, el script los convierte automáticamente a CSV para continuar con el flujo.
2. **SQL/DuckDB** (`src/duckdb_processing.py`): normaliza columnas clave, filtra outliers, crea features derivados y exporta `data/props_model.parquet`. También genera un resumen exploratorio en `reports/duckdb_summary.json`.
3. **Spark** (`src/spark_processing.py`): recalcula ratios, crea variables binarias y limpia nulos, produciendo `data/props_model_sparked.parquet`.
4. **Modelado** (`src/train_models.py`): entrena Regresión Lineal, Árbol de Decisión, Random Forest y Gradient Boosting con un `ColumnTransformer` que preprocesa variables numéricas y categóricas. Evalúa con MAE, RMSE y R², guarda las métricas en `reports/metrics.json`, estadísticas descriptivas en `reports/feature_stats.json`, y persiste el mejor modelo en `models/model.pkl` junto a `models/model_metadata.json`.

### Parámetros útiles

- `--skip-download`: omite la descarga si ya existe un CSV en `data/raw/`.
- `--raw-csv <path>`: permite ejecutar el pipeline con un CSV específico (útil para pruebas con `samples/sample_properties.csv`).

Ejemplo con el CSV sintético incluido:

```bash
python -m src.run_pipeline --raw-csv samples/sample_properties.csv
```

## Aplicación Streamlit

Una vez generado `models/model.pkl`, la aplicación se ejecuta con:

```bash
streamlit run app_streamlit/app.py
```

La app carga el pipeline entrenado, muestra las métricas y permite ingresar características de un inmueble para estimar el precio esperado. Al introducir el precio publicado se emite un dictamen (infravalorada / correcta / sobrevalorada) usando la banda ±10% definida en los metadatos del modelo.

## Archivos clave

- `src/data_download.py`: descarga y organiza el dataset de Kaggle.
- `src/duckdb_processing.py`: limpieza y EDA vía SQL en DuckDB.
- `src/spark_processing.py`: ingeniería de features distribuida en PySpark.
- `src/train_models.py`: entrenamiento y evaluación de los modelos.
- `src/run_pipeline.py`: orquestador del flujo end-to-end.
- `app_streamlit/app.py`: interfaz web para consultar la valuación.

## Desarrollo y pruebas

- Utilizá el CSV sintético `samples/sample_properties.csv` para validar rápidamente que el pipeline y la app funcionan sin descargar el dataset completo.
- Los logs del pipeline se imprimen en consola; habilitá el nivel `DEBUG` si necesitás más detalle.

## Siguientes pasos sugeridos

- Incorporar más variables (amenities, expensas, etc.) y normalizar los textos.
- Experimentar con técnicas de validación cruzada y ajuste de hiperparámetros.
- Desplegar la app en un servicio gestionado (Streamlit Cloud, Heroku, etc.).

