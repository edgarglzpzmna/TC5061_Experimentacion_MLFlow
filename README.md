# TC5061 — Experimentación con MLflow

## Objetivo
Convertir el baseline de DataCo Supply Chain en un script parametrizado y registrar parámetros, métricas, artefactos y modelo en MLflow Tracking.

## Estructura
- `train.py`: carga/preparación de datos, pipeline, entrenamiento, evaluación y registro.
- `requirements.txt`: dependencias con versiones fijadas.
- `data/`: coloca aquí `DataCoSupplyChainDataset.csv`.
- `artifacts/`: resultados locales por corrida.
- `DescriptionDataCoSupplyChain.csv`: diccionario de datos (documentación).

## Preparar entorno
Requiere Python 3.10 o 3.11 recomendado.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# En Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Colocar los datos
Copia `DataCoSupplyChainDataset.csv` en `data/`. El archivo no se incluye automáticamente en el repositorio.

## Iniciar MLflow
En una terminal, desde la raíz del proyecto:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Abre `http://127.0.0.1:8080` en el navegador.

## Ejecutar el baseline
En otra terminal, activa el mismo entorno y ejecuta:

```bash
python train.py --data-path data/DataCoSupplyChainDataset.csv --C 1.0 --class-weight none --max-iter 1000 --random-state 42
```

## Ejecutar una configuración alternativa
```bash
python train.py --data-path data/DataCoSupplyChainDataset.csv --C 0.1 --class-weight balanced --max-iter 1500 --random-state 42
```

Los argumentos `--C`, `--class-weight`, `--max-iter` y `--random-state` permiten variar la configuración sin editar el código. `--experiment-name` agrupa corridas y `--tracking-uri` permite configurar el servidor.

## Métricas y artefactos
Se registran `accuracy`, `precision`, `recall`, `f1_score` y `roc_auc`, además de `metrics.json`, `confusion_matrix.png` y el pipeline entrenado.

## Reproducibilidad y limitaciones
Se fija la semilla del split estratificado y del estimador. La reproducibilidad exacta puede depender del sistema operativo, versiones de bibliotecas, BLAS y recursos de ejecución. El modelo usa un split holdout único; una comparación rigurosa debe mantener el mismo split entre corridas y reconocer que cinco experimentos no demuestran por sí solos causalidad ni significancia estadística.
