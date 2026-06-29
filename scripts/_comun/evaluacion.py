"""Evaluacion compartida para las fases de modelado.

Las tres familias de modelos del proyecto se entrenan en scripts distintos:

    fase 07 (clasicos)      -> SVM y Naive Bayes sobre TF-IDF
    fase 08 (deep learning) -> CNN y LSTM sobre embeddings entrenables
    fase 09 (transformers)  -> BETO y XLM-RoBERTa con fine-tuning

Para que las tres se comparen con EXACTAMENTE el mismo criterio, este modulo centraliza
el calculo de metricas y la generacion de artefactos de evaluacion:

    - F1-Macro (metrica principal, justa con clases minoritarias)
    - exactitud balanceada (balanced accuracy)
    - F1-weighted y accuracy (secundarias)
    - F1 por clase (formato largo)
    - matriz de confusion (CSV + PNG)
    - classification report (txt)

Todas las fases usan el mismo orden de clases (CLASES), por lo que las matrices de
confusion y los reportes son directamente comparables entre familias. El esquema de las
filas de comparacion tambien es comun (familia, modelo, estrategia, split, metricas), de
modo que la fase 10 puede unirlas en una sola tabla sin transformaciones.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana, para guardar PNG en disco
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Orden fijo de clases (escala ordinal de sentimiento). Compartido por todas las fases.
CLASES = ["muy negativo", "negativo", "neutral", "positivo", "muy positivo"]


def metricas_basicas(y_true, y_pred, labels=CLASES):
    """Devuelve el dict de metricas escalares para una prediccion."""
    return {
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def f1_por_clase(y_true, y_pred, labels=CLASES):
    """Devuelve una lista de tuplas (clase, f1) en el orden de labels."""
    valores = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    return [(clase, round(float(valor), 4)) for clase, valor in zip(labels, valores)]


def guardar_matriz_confusion(y_true, y_pred, ruta_dir, combo, titulo=None, labels=CLASES):
    """Guarda la matriz de confusion como CSV y PNG en ruta_dir."""
    ruta_dir = Path(ruta_dir)
    ruta_dir.mkdir(parents=True, exist_ok=True)

    matriz = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matriz, index=labels, columns=labels).to_csv(
        ruta_dir / f"matriz_confusion_{combo}_test.csv", encoding="utf-8-sig"
    )

    display = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(titulo or combo.replace("_", " "))
    fig.tight_layout()
    fig.savefig(ruta_dir / f"matriz_confusion_{combo}_test.png", dpi=120)
    plt.close(fig)


def guardar_reporte_clasificacion(y_true, y_pred, ruta_dir, combo, labels=CLASES):
    """Guarda el classification report de sklearn como .txt en ruta_dir."""
    ruta_dir = Path(ruta_dir)
    ruta_dir.mkdir(parents=True, exist_ok=True)
    reporte = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    (ruta_dir / f"reporte_clasificacion_{combo}_test.txt").write_text(reporte, encoding="utf-8")


def evaluar_split(y_true, y_pred, familia, modelo, estrategia, split, ruta_dir,
                  filas_comparacion, filas_f1_clase, labels=CLASES):
    """Evalua una prediccion y acumula filas para los CSV de comparacion.

    - Agrega una fila a filas_comparacion con las metricas escalares.
    - Agrega filas a filas_f1_clase con el F1 por clase (formato largo).
    - En split == 'test' guarda matriz de confusion (CSV + PNG) y classification report.

    `y_true` e `y_pred` deben ser etiquetas de texto (las mismas de CLASES) para que las
    tres familias produzcan reportes homogeneos. Devuelve el dict de metricas escalares.
    """
    combo = f"{modelo}_{estrategia}"
    metricas = metricas_basicas(y_true, y_pred, labels)
    filas_comparacion.append({
        "familia": familia,
        "modelo": modelo,
        "estrategia": estrategia,
        "split": split,
        **metricas,
    })
    for clase, valor in f1_por_clase(y_true, y_pred, labels):
        filas_f1_clase.append({
            "familia": familia,
            "modelo": modelo,
            "estrategia": estrategia,
            "split": split,
            "clase": clase,
            "f1": valor,
        })

    if split == "test":
        guardar_matriz_confusion(y_true, y_pred, ruta_dir, combo, labels=labels)
        guardar_reporte_clasificacion(y_true, y_pred, ruta_dir, combo, labels=labels)

    return metricas
