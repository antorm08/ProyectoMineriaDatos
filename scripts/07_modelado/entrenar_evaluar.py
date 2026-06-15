"""Fase 07: entrenamiento y evaluacion de clasificadores de sentimiento.

Compara los cuatro algoritmos clasicos que define el documento del proyecto
(Regresion Logistica, SVM, Naive Bayes y Random Forest) sobre TF-IDF con bigramas,
cruzados con tres estrategias frente al desbalance de clases:

    base       -> sin ajuste de balanceo
    balanced   -> class_weight="balanced" (algoritmos que lo soportan)
    smote      -> sobremuestreo SMOTE aplicado SOLO al conjunto de entrenamiento

Naive Bayes no soporta class_weight, por lo que su combinacion con "balanced" se omite.

Metricas (en validacion y prueba):
    - F1-Macro (metrica principal, justa con clases minoritarias)
    - exactitud balanceada (balanced accuracy)
    - F1 por clase
    - matriz de confusion (CSV + PNG, sobre prueba)
    - accuracy (metrica secundaria)

El preprocesamiento TF-IDF elimina stopwords en espanol, pero CONSERVA las palabras
de negacion e intensidad (no, nunca, sin, pero, muy, ...) porque cargan sentimiento.

Entrada:
    data/splits/train.csv, valid.csv, test.csv  (columnas texto_modelo y sentimiento_final)

Salidas (reports/06_modelado/):
    comparacion_modelos.csv                       (todas las combinaciones)
    f1_por_clase.csv                              (F1 por clase, formato largo)
    matriz_confusion_<algoritmo>_<estrategia>_test.csv / .png
    reporte_clasificacion_<algoritmo>_<estrategia>_test.txt
models/:
    mejor_modelo.joblib   (vectorizador + clasificador de la mejor combinacion)
"""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend sin ventana, para guardar PNG en disco
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORT_DIR = PROJECT_ROOT / "reports" / "06_modelado"
MODELS_DIR = PROJECT_ROOT / "models"
MODELO_FILE = MODELS_DIR / "mejor_modelo.joblib"

COLUMNA_TEXTO = "texto_modelo"
COLUMNA_ETIQUETA = "sentimiento_final"

# Orden fijo de clases para que las matrices de confusion sean comparables.
CLASES = ["muy negativo", "negativo", "neutral", "positivo", "muy positivo"]

ALGORITMOS = ["regresion_logistica", "svm", "naive_bayes", "random_forest"]
ESTRATEGIAS = ["base", "balanced", "smote"]
# Naive Bayes no acepta class_weight; el resto si.
SOPORTA_CLASS_WEIGHT = {"regresion_logistica", "svm", "random_forest"}

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


def construir_vectorizador(max_features, usar_stopwords):
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        stop_words=STOPWORDS_ES if usar_stopwords else None,
    )


def construir_clasificador(algoritmo, estrategia, random_state):
    class_weight = "balanced" if estrategia == "balanced" else None
    if algoritmo == "regresion_logistica":
        return LogisticRegression(max_iter=1000, class_weight=class_weight)
    if algoritmo == "svm":
        return LinearSVC(class_weight=class_weight, max_iter=2000)
    if algoritmo == "naive_bayes":
        return MultinomialNB()
    if algoritmo == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Algoritmo desconocido: {algoritmo}")


def aplicar_smote(x_train, y_train, random_state):
    from imblearn.over_sampling import SMOTE

    # k_neighbors no puede superar (n de la clase minoritaria - 1).
    minoritaria = pd.Series(y_train).value_counts().min()
    k = max(1, min(5, minoritaria - 1))
    smote = SMOTE(random_state=random_state, k_neighbors=k)
    return smote.fit_resample(x_train, y_train)


def guardar_matriz_confusion(y_true, y_pred, combo, titulo):
    matriz = confusion_matrix(y_true, y_pred, labels=CLASES)
    pd.DataFrame(matriz, index=CLASES, columns=CLASES).to_csv(
        REPORT_DIR / f"matriz_confusion_{combo}_test.csv", encoding="utf-8-sig"
    )

    display = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=CLASES)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"matriz_confusion_{combo}_test.png", dpi=120)
    plt.close(fig)


def evaluar(clasificador, x, y_true, algoritmo, estrategia, split, filas_f1_clase):
    combo = f"{algoritmo}_{estrategia}"
    y_pred = clasificador.predict(x)

    f1_por_clase = f1_score(y_true, y_pred, average=None, labels=CLASES, zero_division=0)
    for clase, valor in zip(CLASES, f1_por_clase):
        filas_f1_clase.append({
            "algoritmo": algoritmo,
            "estrategia": estrategia,
            "split": split,
            "clase": clase,
            "f1": round(float(valor), 4),
        })

    if split == "test":
        reporte_txt = classification_report(y_true, y_pred, labels=CLASES, zero_division=0)
        (REPORT_DIR / f"reporte_clasificacion_{combo}_test.txt").write_text(reporte_txt, encoding="utf-8")
        guardar_matriz_confusion(y_true, y_pred, combo, combo.replace("_", " "))

    return {
        "algoritmo": algoritmo,
        "estrategia": estrategia,
        "split": split,
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", labels=CLASES, zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", labels=CLASES, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }


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

            clasificador = construir_clasificador(algoritmo, estrategia, random_state)
            clasificador.fit(x_fit, y_fit)

            fila_valid = evaluar(clasificador, x_valid, y_valid, algoritmo, estrategia, "valid", filas_f1_clase)
            fila_test = evaluar(clasificador, x_test, y_test, algoritmo, estrategia, "test", filas_f1_clase)
            filas_comparacion.extend([fila_valid, fila_test])
            modelos[(algoritmo, estrategia)] = {"clasificador": clasificador, "valid": fila_valid, "test": fila_test}

            print(
                f">> {algoritmo:<20} {estrategia:<9} | "
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
    print("COMPARACION DE MODELOS (ordenada por F1-macro en test)")
    print("=" * 78)
    tabla_test = comparacion[comparacion["split"] == "test"].sort_values("f1_macro", ascending=False)
    print(tabla_test.to_string(index=False))
    if omitidas:
        print("\nCombinaciones omitidas:")
        for o in omitidas:
            print(f"- {o}")
    print(f"\nMejor combinacion por F1-macro en validacion: {mejor[0]} + {mejor[1]} "
          f"(valid {modelos[mejor]['valid']['f1_macro']:.4f} | test {modelos[mejor]['test']['f1_macro']:.4f})")
    print(f"Modelo guardado en: {MODELO_FILE}")
    print(f"Reportes en: {REPORT_DIR}")


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Entrena y evalua clasificadores de sentimiento (fase 07).")
    parser.add_argument("--algoritmos", nargs="+", default=ALGORITMOS, choices=ALGORITMOS,
                        help="Algoritmos a comparar.")
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
