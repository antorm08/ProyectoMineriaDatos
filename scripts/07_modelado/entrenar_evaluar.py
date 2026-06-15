"""Fase 07: entrenamiento y evaluacion del clasificador de sentimiento.

Entrena un clasificador de texto (TF-IDF + Regresion Logistica) sobre los splits
generados en la fase 06 y compara tres estrategias frente al desbalance de clases:

    base       -> sin ningun ajuste de balanceo
    balanced   -> LogisticRegression(class_weight="balanced")
    smote      -> sobremuestreo SMOTE aplicado SOLO al conjunto de entrenamiento

Para cada estrategia reporta, en validacion y prueba:
    - F1-Macro (metrica principal, justa con clases minoritarias)
    - F1 por clase
    - matriz de confusion (CSV + PNG)
    - accuracy (metrica secundaria)

Entrada:
    data/splits/train.csv, valid.csv, test.csv  (columnas texto_modelo y sentimiento_final)

Salidas (reports/06_modelado/):
    comparacion_modelos.csv
    f1_por_clase_<estrategia>.csv
    matriz_confusion_<estrategia>_<split>.csv / .png
    reporte_clasificacion_<estrategia>_<split>.txt
models/:
    modelo_<mejor_estrategia>.joblib   (vectorizador + clasificador)

Depende de pandas, scikit-learn, imbalanced-learn (SMOTE), joblib y matplotlib.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend sin ventana, para guardar PNG en disco
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_DIR = PROJECT_ROOT / "reports" / "06_modelado"
MODELS_DIR = PROJECT_ROOT / "models"

COLUMNA_TEXTO = "texto_modelo"
COLUMNA_ETIQUETA = "sentimiento_final"

# Orden fijo de clases para que las matrices de confusion sean comparables.
CLASES = ["muy negativo", "negativo", "neutral", "positivo", "muy positivo"]


def cargar_split(nombre):
    ruta = SPLITS_DIR / f"{nombre}.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el split {nombre}: {ruta}. Corre primero la fase 06.")

    df = pd.read_csv(ruta).fillna("")
    faltantes = {COLUMNA_TEXTO, COLUMNA_ETIQUETA} - set(df.columns)
    if faltantes:
        raise ValueError(f"El split {nombre} no tiene columnas requeridas: {sorted(faltantes)}")

    df = df[(df[COLUMNA_TEXTO].astype(str).str.strip() != "") & (df[COLUMNA_ETIQUETA].astype(str).str.strip() != "")]
    return df[COLUMNA_TEXTO].astype(str), df[COLUMNA_ETIQUETA].astype(str)


def construir_vectorizador(max_features):
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )


def construir_clasificador(estrategia):
    if estrategia == "balanced":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    return LogisticRegression(max_iter=1000)


def aplicar_smote(x_train, y_train, random_state):
    from imblearn.over_sampling import SMOTE

    # k_neighbors no puede superar (n de la clase minoritaria - 1).
    minoritaria = pd.Series(y_train).value_counts().min()
    k = max(1, min(5, minoritaria - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    return smote.fit_resample(x_train, y_train)


def guardar_matriz_confusion(y_true, y_pred, estrategia, split, titulo):
    matriz = confusion_matrix(y_true, y_pred, labels=CLASES)
    matriz_df = pd.DataFrame(matriz, index=CLASES, columns=CLASES)
    matriz_df.to_csv(REPORT_DIR / f"matriz_confusion_{estrategia}_{split}.csv", encoding="utf-8-sig")

    display = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=CLASES)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"matriz_confusion_{estrategia}_{split}.png", dpi=120)
    plt.close(fig)


def evaluar(clasificador, x, y_true, estrategia, split):
    y_pred = clasificador.predict(x)
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=CLASES, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=CLASES, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # F1 por clase.
    f1_por_clase = f1_score(y_true, y_pred, average=None, labels=CLASES, zero_division=0)
    f1_clase_df = pd.DataFrame({"clase": CLASES, "f1": f1_por_clase.round(4)})
    f1_clase_df.to_csv(REPORT_DIR / f"f1_por_clase_{estrategia}_{split}.csv", index=False, encoding="utf-8-sig")

    # Reporte de clasificacion completo.
    reporte_txt = classification_report(y_true, y_pred, labels=CLASES, zero_division=0)
    (REPORT_DIR / f"reporte_clasificacion_{estrategia}_{split}.txt").write_text(reporte_txt, encoding="utf-8")

    guardar_matriz_confusion(y_true, y_pred, estrategia, split, f"{estrategia} - {split}")

    return {
        "estrategia": estrategia,
        "split": split,
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "accuracy": round(accuracy, 4),
    }


def entrenar_y_evaluar(estrategias, max_features, random_state):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x_train_txt, y_train = cargar_split("train")
    x_valid_txt, y_valid = cargar_split("valid")
    x_test_txt, y_test = cargar_split("test")

    vectorizador = construir_vectorizador(max_features)
    x_train = vectorizador.fit_transform(x_train_txt)
    x_valid = vectorizador.transform(x_valid_txt)
    x_test = vectorizador.transform(x_test_txt)
    print(f"TF-IDF: {x_train.shape[0]} train x {x_train.shape[1]} features")

    filas_comparacion = []
    resultados_por_estrategia = {}

    for estrategia in estrategias:
        print(f"\n>> Entrenando estrategia: {estrategia}")
        x_fit, y_fit = x_train, y_train
        if estrategia == "smote":
            x_fit, y_fit = aplicar_smote(x_train, y_train, random_state)
            print(f"   SMOTE: train pasa de {x_train.shape[0]} a {x_fit.shape[0]} filas (clases balanceadas)")

        clasificador = construir_clasificador(estrategia)
        clasificador.fit(x_fit, y_fit)

        fila_valid = evaluar(clasificador, x_valid, y_valid, estrategia, "valid")
        fila_test = evaluar(clasificador, x_test, y_test, estrategia, "test")
        filas_comparacion.extend([fila_valid, fila_test])
        resultados_por_estrategia[estrategia] = {"clasificador": clasificador, "test": fila_test}

        print(f"   valid -> F1-macro {fila_valid['f1_macro']:.4f} | accuracy {fila_valid['accuracy']:.4f}")
        print(f"   test  -> F1-macro {fila_test['f1_macro']:.4f} | accuracy {fila_test['accuracy']:.4f}")

    comparacion = pd.DataFrame(filas_comparacion)
    comparacion.to_csv(REPORT_DIR / "comparacion_modelos.csv", index=False, encoding="utf-8-sig")

    # La mejor estrategia se elige por F1-macro en validacion (no en test, para no sesgar).
    f1_valid = {
        estrategia: next(f["f1_macro"] for f in filas_comparacion if f["estrategia"] == estrategia and f["split"] == "valid")
        for estrategia in estrategias
    }
    mejor = max(f1_valid, key=f1_valid.get)
    modelo = {"vectorizador": vectorizador, "clasificador": resultados_por_estrategia[mejor]["clasificador"], "clases": CLASES, "estrategia": mejor}
    ruta_modelo = MODELS_DIR / f"modelo_{mejor}.joblib"
    joblib.dump(modelo, ruta_modelo)

    print("\n" + "=" * 70)
    print("COMPARACION DE MODELOS")
    print("=" * 70)
    print(comparacion.to_string(index=False))
    print(f"\nMejor estrategia por F1-macro en validacion: {mejor} (F1-macro valid {f1_valid[mejor]:.4f})")
    print(f"F1-macro en test de la mejor estrategia: {resultados_por_estrategia[mejor]['test']['f1_macro']:.4f}")
    print(f"\nModelo guardado en: {ruta_modelo}")
    print(f"Reportes en: {REPORT_DIR}")


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Entrena y evalua el clasificador de sentimiento (fase 07).")
    parser.add_argument(
        "--estrategias",
        nargs="+",
        default=["base", "balanced", "smote"],
        choices=["base", "balanced", "smote"],
        help="Estrategias de balanceo a comparar.",
    )
    parser.add_argument("--max-features", type=int, default=20000, help="Maximo de features TF-IDF.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    entrenar_y_evaluar(args.estrategias, args.max_features, args.random_state)


if __name__ == "__main__":
    main()
