"""Entrena el modelo etiquetador inicial con la muestra manual.

Usa validacion cruzada sobre los 500 registros etiquetados manualmente para
comparar SVM y Naive Bayes. Luego reentrena el mejor modelo con los 500 completos
y lo guarda para etiquetar el resto del 80% de desarrollo.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
from _comun.evaluacion import CLASES  # noqa: E402

INPUT_FILE = PROJECT_ROOT / "data" / "manual_500" / "muestra_500_para_etiquetar.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "11_etiquetado_manual"
MODEL_FILE = PROJECT_ROOT / "models" / "modelo_etiquetador_inicial.joblib"


def construir_modelos(max_features):
    vectorizador = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
    )
    return {
        "svm_balanced": Pipeline([
            ("tfidf", vectorizador),
            ("clf", LinearSVC(class_weight="balanced", max_iter=2000)),
        ]),
        "naive_bayes": Pipeline([
            ("tfidf", vectorizador),
            ("clf", MultinomialNB()),
        ]),
    }


def validar_etiquetas(df):
    if "etiqueta_manual" not in df.columns:
        raise ValueError("Falta la columna 'etiqueta_manual'.")
    df = df[df["etiqueta_manual"].astype(str).str.strip() != ""].copy()
    invalidas = sorted(set(df["etiqueta_manual"]) - set(CLASES))
    if invalidas:
        raise ValueError(f"Etiquetas manuales invalidas: {invalidas}. Usa: {CLASES}")
    if len(df) < 50:
        raise ValueError("Hay muy pocas filas etiquetadas. Completa la muestra manual antes de entrenar.")
    return df


def entrenar(input_file, report_dir, model_file, folds, max_features, random_state):
    report_dir.mkdir(parents=True, exist_ok=True)
    model_file.parent.mkdir(parents=True, exist_ok=True)

    df = validar_etiquetas(pd.read_csv(input_file).fillna(""))
    x = df["texto_modelo"].astype(str)
    y = df["etiqueta_manual"].astype(str)

    conteos = y.value_counts()
    folds_efectivos = min(folds, int(conteos.min()))
    if folds_efectivos < 2:
        raise ValueError("Cada clase debe tener al menos 2 ejemplos para validacion cruzada.")

    cv = StratifiedKFold(n_splits=folds_efectivos, shuffle=True, random_state=random_state)
    filas = []
    matrices = []
    modelos = construir_modelos(max_features)

    for nombre, modelo in modelos.items():
        pred = cross_val_predict(modelo, x, y, cv=cv)
        f1_macro = f1_score(y, pred, average="macro", labels=CLASES, zero_division=0)
        accuracy = accuracy_score(y, pred)
        filas.append({"modelo": nombre, "folds": folds_efectivos, "f1_macro_cv": round(f1_macro, 4), "accuracy_cv": round(accuracy, 4)})
        matriz = pd.DataFrame(confusion_matrix(y, pred, labels=CLASES), index=CLASES, columns=CLASES)
        matriz.insert(0, "modelo", nombre)
        matrices.append(matriz.reset_index(names="real"))

    comparacion = pd.DataFrame(filas).sort_values("f1_macro_cv", ascending=False)
    comparacion.to_csv(report_dir / "comparacion_modelo_inicial_cv.csv", index=False, encoding="utf-8-sig")
    pd.concat(matrices, ignore_index=True).to_csv(report_dir / "matrices_confusion_modelo_inicial_cv.csv", index=False, encoding="utf-8-sig")

    mejor_nombre = comparacion.iloc[0]["modelo"]
    mejor_modelo = modelos[mejor_nombre]
    mejor_modelo.fit(x, y)
    joblib.dump({"modelo": mejor_modelo, "clases": CLASES, "nombre": mejor_nombre}, model_file)

    hiperparametros = {
        "tfidf": {"ngram_range": [1, 2], "min_df": 2, "max_features": max_features, "sublinear_tf": True},
        "svm_balanced": {"class_weight": "balanced", "max_iter": 2000},
        "naive_bayes": {"modelo": "MultinomialNB", "parametros_default_sklearn": True},
        "validacion_cruzada": {"folds": folds_efectivos, "shuffle": True, "random_state": random_state},
    }
    (report_dir / "hiperparametros_modelo_inicial.json").write_text(json.dumps(hiperparametros, indent=2), encoding="utf-8")

    print(comparacion.to_string(index=False))
    print(f"Mejor modelo inicial: {mejor_nombre}")
    print(f"Modelo reentrenado con la muestra completa guardado en: {model_file}")


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Entrena el etiquetador inicial con validacion cruzada.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--model-file", type=Path, default=MODEL_FILE)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    try:
        entrenar(args.input, args.report_dir, args.model_file, args.folds, args.max_features, args.random_state)
    except (FileNotFoundError, ValueError) as exc:
        print(f"No se puede entrenar el modelo inicial: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
