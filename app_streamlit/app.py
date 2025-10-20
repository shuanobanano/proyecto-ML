from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FEATURE_STATS_PATH = PROJECT_ROOT / "reports" / "feature_stats.json"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model was not found. Run `python -m src.run_pipeline` first."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata() -> Dict:
    if not MODEL_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Model metadata missing. Ensure the training step completed successfully."
        )
    return json.loads(MODEL_METADATA_PATH.read_text())


@st.cache_data
def load_feature_stats() -> Dict:
    if FEATURE_STATS_PATH.exists():
        return json.loads(FEATURE_STATS_PATH.read_text())
    return {}


@st.cache_data
def load_metrics() -> Dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def compute_auxiliary_features(row: Dict[str, float]) -> Dict[str, float]:
    surface = row.get("surface_total_m2", 0)
    covered = row.get("surface_covered_m2", 0)
    rooms = row.get("rooms", 0)
    if surface and surface > 0:
        row["covered_ratio"] = covered / surface if covered else 0
        row["rooms_per_surface"] = rooms / surface if rooms else 0
    else:
        row["covered_ratio"] = 0
        row["rooms_per_surface"] = 0
    return row


def classify_price(predicted: float, actual: float, band: float) -> str:
    lower = predicted * (1 - band)
    upper = predicted * (1 + band)
    if actual < lower:
        return "Infravalorada"
    if actual > upper:
        return "Sobrevalorada"
    return "Correcta"


def main() -> None:
    st.set_page_config(page_title="Valuador inmobiliario CABA", page_icon="🏙️")
    st.title("Valuador inmobiliario con ML")
    st.write(
        """
        Este MVP descarga datos de propiedades en CABA, limpia los registros con SQL sobre DuckDB,
        enriquece variables con PySpark y entrena múltiples modelos (Regresión Lineal, Árboles de Decisión,
        Random Forest y Gradient Boosting). Aquí podés estimar el precio esperado de una propiedad y evaluar
        si un aviso está infravalorado, correctamente valuado o sobrevaluado considerando una banda ±10%.
        """
    )

    metadata = load_metadata()
    feature_stats = load_feature_stats()
    metrics = load_metrics()

    with st.expander("📈 Métricas de modelos"):
        if metrics:
            df_metrics = pd.DataFrame(metrics).drop(columns=["best_model", "best_model_metrics"], errors="ignore")
            st.dataframe(df_metrics.T)
            st.write(
                f"Modelo seleccionado: **{metrics.get('best_model', 'N/A')}** con RMSE "
                f"{metrics.get('best_model_metrics', {}).get('rmse', 'N/A')}"
            )
        else:
            st.info("Ejecutá la canalización para generar las métricas.")

    st.header("Ingresá las características de la propiedad")

    numeric_inputs = {}
    categorical_inputs = {}

    cols = st.columns(2)

    with cols[0]:
        surface_total = st.number_input(
            "Superficie total (m²)",
            min_value=10.0,
            max_value=float(feature_stats.get("surface_total_m2", {}).get("max", 300.0)),
            value=float(feature_stats.get("surface_total_m2", {}).get("median", 60.0)),
        )
        surface_covered = st.number_input(
            "Superficie cubierta (m²)",
            min_value=0.0,
            max_value=float(feature_stats.get("surface_covered_m2", {}).get("max", surface_total)),
            value=min(
                surface_total,
                float(feature_stats.get("surface_covered_m2", {}).get("median", surface_total * 0.9)),
            ),
        )
        rooms = st.number_input(
            "Ambientes",
            min_value=1.0,
            max_value=float(feature_stats.get("rooms", {}).get("max", 6.0)),
            value=float(feature_stats.get("rooms", {}).get("median", 3.0)),
        )
        bedrooms = st.number_input(
            "Dormitorios",
            min_value=0.0,
            max_value=float(feature_stats.get("bedrooms", {}).get("max", rooms)),
            value=min(
                rooms,
                float(feature_stats.get("bedrooms", {}).get("median", max(1.0, rooms - 1))),
            ),
        )
        bathrooms = st.number_input(
            "Baños",
            min_value=1.0,
            max_value=float(feature_stats.get("bathrooms", {}).get("max", 3.0)),
            value=float(feature_stats.get("bathrooms", {}).get("median", 1.0)),
        )
    with cols[1]:
        latitude = st.number_input(
            "Latitud",
            min_value=float(feature_stats.get("latitude", {}).get("min", -34.7)),
            max_value=float(feature_stats.get("latitude", {}).get("max", -34.5)),
            value=float(feature_stats.get("latitude", {}).get("median", -34.6)),
            format="%.5f",
        )
        longitude = st.number_input(
            "Longitud",
            min_value=float(feature_stats.get("longitude", {}).get("min", -58.6)),
            max_value=float(feature_stats.get("longitude", {}).get("max", -58.3)),
            value=float(feature_stats.get("longitude", {}).get("median", -58.45)),
            format="%.5f",
        )
        property_age = st.number_input(
            "Antigüedad (años)",
            min_value=0.0,
            max_value=float(feature_stats.get("property_age", {}).get("max", 80.0)),
            value=float(feature_stats.get("property_age", {}).get("median", 20.0)),
        )

    numeric_inputs.update(
        {
            "surface_total_m2": surface_total,
            "surface_covered_m2": surface_covered,
            "rooms": rooms,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "latitude": latitude,
            "longitude": longitude,
            "property_age": property_age,
        }
    )

    categorical_features = metadata.get("categorical_features", [])

    property_type_choices = feature_stats.get("property_type", {}).get(
        "top_values", ["Departamento", "Casa", "PH"]
    )
    state_name_choices = feature_stats.get("state_name", {}).get("top_values", ["CABA"])
    neighborhood_choices = feature_stats.get("neighborhood", {}).get("top_values", ["Palermo", "Belgrano"])

    categorical_inputs["property_type"] = st.selectbox(
        "Tipo de propiedad", property_type_choices, index=0
    )
    categorical_inputs["state_name"] = st.selectbox(
        "Comuna/Provincia", state_name_choices, index=0
    )
    categorical_inputs["neighborhood"] = st.selectbox(
        "Barrio", neighborhood_choices, index=0
    )

    listed_price = st.number_input(
        "Precio publicado (USD)",
        min_value=0.0,
        value=0.0,
        help="Ingresalo si querés comparar contra la valuación del modelo",
    )

    if st.button("Calcular valuación"):
        model = load_model()
        band = metadata.get("band_percentage", 0.10)

        features = {**numeric_inputs, **categorical_inputs}
        features = compute_auxiliary_features(features)

        for column in metadata.get("numeric_features", []):
            features.setdefault(column, 0)
        for column in categorical_features:
            features.setdefault(column, "Desconocido")

        input_df = pd.DataFrame([features])
        prediction = float(model.predict(input_df)[0])

        st.success(f"Precio estimado: USD {prediction:,.0f}")
        st.write(
            f"Precio por m² estimado: USD {prediction / features['surface_total_m2'] if features['surface_total_m2'] else 0:,.0f}"
        )

        if listed_price > 0:
            classification = classify_price(prediction, listed_price, band)
            st.write(
                f"El aviso ingresado está **{classification}** (banda ±{band * 100:.0f}%)."
            )
        else:
            st.info("Ingresá un precio publicado para calcular el dictamen.")

    st.write("---")
    st.caption(
        "Para volver a ejecutar el pipeline completo usá `python -m src.run_pipeline`."
    )


if __name__ == "__main__":
    main()

