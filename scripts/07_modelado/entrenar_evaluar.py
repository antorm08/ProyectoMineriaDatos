"""Fase 07: entrenamiento y evaluacion de los modelos CLASICOS.

Esta fase cubre la familia de modelos clasicos del proyecto: **SVM** (LinearSVC) y
**Naive Bayes** (MultinomialNB) sobre una representacion TF-IDF con bigramas. Ambos se
cruzan con tres estrategias frente al desbalance de clases:

    base       -> sin ajuste de balanceo
    balanced   -> class_weight="balanced" (solo SVM; Naive Bayes no lo soporta)
    smote      -> sobremuestreo SMOTE aplicado SOLO al conjunto de entrenamiento

Las otras familias viven en fases separadas y comparten el modulo de evaluacion:

    fase 08 (scripts/08_dl)           -> CNN y LSTM (deep learning)
    fase 09 (scripts/09_transformers) -> BETO y XLM-RoBERTa (transformers)
    fase 10 (scripts/10_comparacion)  -> comparacion unificada de las tres familias

Metricas (en validacion y prueba): F1-Macro (principal), exactitud balanceada,
F1-weighted, accuracy, F1 por clase, matriz de confusion (CSV + PNG) y classification
report. El calculo se delega a scripts/_comun/evaluacion.py para que las tres familias
se midan igual.

El preprocesamiento TF-IDF elimina stopwords en espanol, pero CONSERVA las palabras de
negacion e intensidad (no, nunca, sin, pero, muy, ...) porque cargan sentimiento.

Entrada:
    data/splits/train.csv, valid.csv, test.csv  (columnas texto_modelo y sentimiento_final)

Salidas (reports/06_modelado/):
    comparacion_modelos.csv                       (todas las combinaciones)
    f1_por_clase.csv                              (F1 por clase, formato largo)
    matriz_confusion_<modelo>_<estrategia>_test.csv / .png
    reporte_clasificacion_<modelo>_<estrategia>_test.txt
models/:
    mejor_modelo.joblib   (vectorizador + clasificador de la mejor combinacion clasica)
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
from _comun.datos import cargar_split  # noqa: E402
from _comun.evaluacion import CLASES, evaluar_split  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "06_modelado"
MODELS_DIR = PROJECT_ROOT / "models"
MODELO_FILE = MODELS_DIR / "mejor_modelo.joblib"

FAMILIA = "clasico"

# Solo los dos clasicos del enunciado: SVM y Naive Bayes.
ALGORITMOS = ["svm", "naive_bayes"]
ESTRATEGIAS = ["base", "balanced", "smote"]
# Naive Bayes no acepta class_weight; SVM (LinearSVC) si.
SOPORTA_CLASS_WEIGHT = {"svm"}

# Palabras de negacion/intensidad que SI se conservan (cargan sentimiento).
PALABRAS_SENTIMIENTO = {
    "no", "ni", "nunca", "jamas", "nada", "nadie", "ningun", "ninguno", "ninguna",
    "sin", "pero", "tampoco", "aunque", "sino", "muy", "mas", "menos", "mucho",
    "poco", "tan", "demasiado", "bien", "mal",
}

# Stopwords en espanol (lista curada de palabras funcionales frecuentes).
STOPWORDS_ES_BASE = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
    "para", "con", "una", "su", "al", "lo", "como", "sus", "le", "ya", "o", "este",
    "si", "porque", "esta", "entre", "cuando", "sobre", "tambien", "me", "hasta",
    "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos", "uno", "les",
    "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto", "antes", "algunos",
    "que", "unos", "yo", "otro", "otras", "otra", "el", "esa", "estos", "quienes",
    "muchos", "cual", "ella", "estar", "estas", "algunas", "nosotros", "mi", "mis",
    "tu", "te", "ti", "tus", "ellas", "os", "mio", "mia", "fue", "ha", "han", "he",
    "ser", "es", "son", "era", "soy", "estoy", "esta", "estan", "fui", "del", "y",
    "the", "of", "to", "este", "esta", "esto", "aqui", "alli", "asi", "cada", "vez",
    "solo", "luego", "despues", "entonces", "mismo", "misma", "tener", "hacer", "ir",
}

# Stopwords finales: base menos las palabras de sentimiento que queremos conservar.
STOPWORDS_ES = sorted(STOPWORDS_ES_BASE - PALABRAS_SENTIMIENTO)


def construir_vectorizador(max_features, usar_stopwords):
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        stop_words=STOPWORDS_ES if usar_stopwords else None,
    )


def construir_clasificador(algoritmo, estrategia):
    class_weight = "balanced" if estrategia == "balanced" else None
    if algoritmo == "svm":
        return LinearSVC(class_weight=class_weight, max_iter=2000)
    if algoritmo == "naive_bayes":
        return MultinomialNB()
    raise ValueError(f"Algoritmo desconocido: {algoritmo}")


def aplicar_smote(x_train, y_train, random_state):
    from imblearn.over_sampling import SMOTE

    # k_neighbors no puede superar (n de la clase minoritaria - 1).
    minoritaria = pd.Series(y_train).value_counts().min()
    k = max(1, min(5, minoritaria - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    return smote.fit_resample(x_train, y_train)


def obtener_scores(clasificador, x):
    """Devuelve probabilidades si existen; si no, scores de decision para ROC/AUC."""
    if hasattr(clasificador, "predict_proba"):
        return clasificador.predict_proba(x), list(clasificador.classes_)
    if hasattr(clasificador, "decision_function"):
        return clasificador.decision_function(x), list(clasificador.classes_)
    return None, None


def entrenar_y_evaluar(algoritmos, estrategias, max_features, random_state, usar_stopwords):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x_train_txt, y_train = cargar_split("train")
    x_valid_txt, y_valid = cargar_split("valid")
    x_test_txt, y_test = cargar_split("test")

    vectorizador = construir_vectorizador(max_features, usar_stopwords)
    x_train = vectorizador.fit_transform(x_train_txt)
    x_valid = vectorizador.transform(x_valid_txt)
    x_test = vectorizador.transform(x_test_txt)
    print(f"TF-IDF: {x_train.shape[0]} train x {x_train.shape[1]} features (stopwords ES: {'si' if usar_stopwords else 'no'})")

    filas_comparacion = []
    filas_f1_clase = []
    modelos = {}
    omitidas = []

    for algoritmo in algoritmos:
        for estrategia in estrategias:
            if estrategia == "balanced" and algoritmo not in SOPORTA_CLASS_WEIGHT:
                omitidas.append(f"{algoritmo}_{estrategia} (no soporta class_weight)")
                continue

            x_fit, y_fit = x_train, y_train
            if estrategia == "smote":
                x_fit, y_fit = aplicar_smote(x_train, y_train, random_state)

            clasificador = construir_clasificador(algoritmo, estrategia)
            clasificador.fit(x_fit, y_fit)

            fila_valid = evaluar_split(
                y_valid, clasificador.predict(x_valid), FAMILIA, algoritmo, estrategia,
                "valid", REPORT_DIR, filas_comparacion, filas_f1_clase,
            )
            pred_test = clasificador.predict(x_test)
            scores_test, score_labels = obtener_scores(clasificador, x_test)
            fila_test = evaluar_split(
                y_test, pred_test, FAMILIA, algoritmo, estrategia,
                "test", REPORT_DIR, filas_comparacion, filas_f1_clase,
                scores=scores_test, score_labels=score_labels,
            )
            modelos[(algoritmo, estrategia)] = {"clasificador": clasificador, "valid": fila_valid, "test": fila_test}

            print(
                f">> {algoritmo:<14} {estrategia:<9} | "
                f"valid F1-macro {fila_valid['f1_macro']:.4f} | test F1-macro {fila_test['f1_macro']:.4f}"
            )

    comparacion = pd.DataFrame(filas_comparacion)
    comparacion.to_csv(REPORT_DIR / "comparacion_modelos.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_f1_clase).to_csv(REPORT_DIR / "f1_por_clase.csv", index=False, encoding="utf-8-sig")

    # La mejor combinacion se elige por F1-macro en validacion (no en test, para no sesgar).
    mejor = max(modelos, key=lambda combo: modelos[combo]["valid"]["f1_macro"])
    modelo = {
        "vectorizador": vectorizador,
        "clasificador": modelos[mejor]["clasificador"],
        "clases": CLASES,
        "algoritmo": mejor[0],
        "estrategia": mejor[1],
    }
    joblib.dump(modelo, MODELO_FILE)

    print("\n" + "=" * 78)
    print("COMPARACION DE MODELOS CLASICOS (ordenada por F1-macro en test)")
    print("=" * 78)
    tabla_test = comparacion[comparacion["split"] == "test"].sort_values("f1_macro", ascending=False)
    print(tabla_test.to_string(index=False))
    if omitidas:
        print("\nCombinaciones omitidas:")
        for o in omitidas:
            print(f"- {o}")
    print(f"\nMejor combinacion clasica por F1-macro en validacion: {mejor[0]} + {mejor[1]} "
          f"(valid {modelos[mejor]['valid']['f1_macro']:.4f} | test {modelos[mejor]['test']['f1_macro']:.4f})")
    print(f"Modelo guardado en: {MODELO_FILE}")
    print(f"Reportes en: {REPORT_DIR}")


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Entrena y evalua los modelos clasicos (SVM y Naive Bayes, fase 07).")
    parser.add_argument("--algoritmos", nargs="+", default=ALGORITMOS, choices=ALGORITMOS,
                        help="Algoritmos clasicos a comparar.")
    parser.add_argument("--estrategias", nargs="+", default=ESTRATEGIAS, choices=ESTRATEGIAS,
                        help="Estrategias de balanceo a comparar.")
    parser.add_argument("--max-features", type=int, default=20000, help="Maximo de features TF-IDF.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sin-stopwords", action="store_true",
                        help="No elimina stopwords en espanol del TF-IDF.")
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    entrenar_y_evaluar(
        args.algoritmos,
        args.estrategias,
        args.max_features,
        args.random_state,
        usar_stopwords=not args.sin_stopwords,
    )


if __name__ == "__main__":
    main()
