"""
Script de entrenamiento reproducible para DataCo Supply Chain.

Ejemplo:
python train.py --data-path data/DataCoSupplyChainDataset.csv --C 1.0 \
  --class-weight balanced --max-iter 1000 --random-state 42

El script registra parámetros, métricas, artefactos y el modelo en MLflow.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Late_delivery_risk"

NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "Product Price",
    "Order Item Quantity",
    "Order Item Discount",
    "Order Item Discount Rate",
    "order_year",
    "order_month",
    "order_dayofweek",
    "order_hour",
]

CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Customer Segment",
    "Category Name",
    "Department Name",
]

# Se conserva el criterio de limpieza del notebook original.
LEAKAGE_COLUMNS = [
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
]
PERSONAL_COLUMNS = [
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
]
ID_COLUMNS = [
    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",
    "Product Card Id",
    "Product Category Id",
    "Category Id",
    "Department Id",
    "Order Item Cardprod Id",
]
HIGH_MISSING_COLUMNS = ["Product Description", "Order Zipcode"]


def parse_args() -> argparse.Namespace:
    """Lee argumentos de ejecución y valida sus rangos."""
    parser = argparse.ArgumentParser(
        description=(
            "Entrena una regresión logística para predecir Late_delivery_risk "
            "y registra cada corrida en MLflow."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/DataCoSupplyChainDataset.csv"),
        help="Ruta al CSV de DataCo.",
    )
    parser.add_argument(
        "--C", type=float, default=1.0,
        help="Inverso de la fuerza de regularización; debe ser > 0 (default: 1.0).",
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Ponderación de clases (default: none).",
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000,
        help="Máximo de iteraciones del optimizador (default: 1000).",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Semilla para la partición y el modelo (default: 42).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.20,
        help="Proporción de datos reservada para prueba (default: 0.20).",
    )
    parser.add_argument(
        "--experiment-name",
        default="TC5061 - DataCo Supply Chain - MLflow Experiments",
        help="Nombre del experimento MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:8080",
        help="URI del servidor MLflow Tracking (default: http://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts"),
        help="Directorio local para guardar artefactos (default: artifacts/).",
    )
    args = parser.parse_args()

    if args.C <= 0:
        parser.error("--C debe ser mayor que cero.")
    if args.max_iter < 1:
        parser.error("--max-iter debe ser al menos 1.")
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar entre 0 y 1.")
    return args


def load_and_prepare_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el CSV y reproduce la limpieza y creación de variables del notebook."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {data_path}. "
            "Coloca el CSV en data/ o utiliza --data-path."
        )

    df = pd.read_csv(data_path, encoding="latin-1", low_memory=False)

    required = [TARGET, "order date (DateOrders)"] + NUMERIC_FEATURES[:5] + CATEGORICAL_FEATURES
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")

    # Crear características temporales a partir de la fecha del pedido.
    order_date = pd.to_datetime(
        df["order date (DateOrders)"], errors="coerce"
    )
    df["order_year"] = order_date.dt.year
    df["order_month"] = order_date.dt.month
    df["order_dayofweek"] = order_date.dt.dayofweek
    df["order_hour"] = order_date.dt.hour

    # Eliminar variables con fuga de información, datos personales,
    # identificadores y columnas con alta proporción de faltantes,
    # siguiendo el notebook de baseline.
    columns_to_drop = (
        LEAKAGE_COLUMNS + PERSONAL_COLUMNS + ID_COLUMNS + HIGH_MISSING_COLUMNS
    )
    df = df.drop(
        columns=[column for column in columns_to_drop if column in df.columns],
        errors="ignore",
    )

    model_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    model_df = df[model_columns].copy()
    model_df = model_df.dropna(subset=[TARGET])

    # El objetivo debe ser binario numérico.
    model_df[TARGET] = pd.to_numeric(model_df[TARGET], errors="coerce")
    model_df = model_df.dropna(subset=[TARGET])
    model_df[TARGET] = model_df[TARGET].astype(int)

    invalid_labels = set(model_df[TARGET].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"Se esperaban etiquetas 0/1; se encontraron: {invalid_labels}")

    X = model_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = model_df[TARGET]
    if y.nunique() != 2:
        raise ValueError("La variable objetivo debe contener las dos clases 0 y 1.")

    return X, y


def build_pipeline(C: float, class_weight: str, max_iter: int, random_state: int) -> Pipeline:
    """Construye el preprocesamiento y clasificador en un solo pipeline."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    classifier = LogisticRegression(
        C=C,
        class_weight=None if class_weight == "none" else "balanced",
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, object]:
    """Calcula métricas de evaluación y la matriz de confusión."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    return metrics, confusion_matrix(y_test, predictions, labels=[0, 1])


def save_artifacts(metrics: dict, matrix, output_dir: Path) -> list[Path]:
    """Guarda métricas JSON y matriz de confusión PNG como artefactos."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    matrix_path = output_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No riesgo (0)", "Riesgo (1)"],
    ).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Matriz de confusión - DataCo baseline")
    fig.tight_layout()
    fig.savefig(matrix_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [metrics_path, matrix_path]


def run_experiment(args: argparse.Namespace) -> dict:
    """Ejecuta una corrida y registra sus resultados en MLflow."""
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    X, y = load_and_prepare_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_pipeline(
        C=args.C,
        class_weight=args.class_weight,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )

    with mlflow.start_run(run_name=f"LogReg-C{args.C}-cw-{args.class_weight}-iter{args.max_iter}") as run:
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "C": args.C,
                "class_weight": args.class_weight,
                "max_iter": args.max_iter,
                "random_state": args.random_state,
                "test_size": args.test_size,
                "split_strategy": "stratified",
                "numeric_feature_count": len(NUMERIC_FEATURES),
                "categorical_feature_count": len(CATEGORICAL_FEATURES),
                "feature_count": len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X_train, y_train)
        for warning_index, warning in enumerate(caught, start=1):
            mlflow.log_param(f"fit_warning_{warning_index}", str(warning.message)[:240])

        metrics, matrix = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        run_output_dir = args.output_dir / run.info.run_id
        artifact_paths = save_artifacts(metrics, matrix, run_output_dir)
        for artifact_path in artifact_paths:
            mlflow.log_artifact(str(artifact_path), artifact_path="evaluation")

        # Compatible con versiones de MLflow que utilizan artifact_path.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5),
        )

        result = {
            "run_id": run.info.run_id,
            "experiment_name": args.experiment_name,
            "metrics": metrics,
            "artifacts": [str(path) for path in artifact_paths],
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
