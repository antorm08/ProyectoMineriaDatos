"""Fase 10: comparacion unificada de las tres familias de modelos.

Reune las comparaciones individuales generadas por cada fase de modelado y produce una
sola tabla y un grafico para ver, de un vistazo, como rinden los seis modelos del
proyecto sobre el mismo conjunto de prueba:

    clasicos     (fase 07) -> SVM, Naive Bayes
    deep learning (fase 08) -> CNN, LSTM
    transformers  (fase 09) -> BETO, XLM-RoBERTa

Todas las fases escriben su comparacion con el mismo esquema de columnas
(familia, modelo, estrategia, split, f1_macro, balanced_accuracy, f1_weighted, accuracy),
gracias a scripts/_comun/evaluacion.py, asi que aqui solo hay que concatenar.

Entrada:
    reports/06_modelado/comparacion_modelos.csv        (clasicos)
    reports/07_dl/comparacion_dl.csv                   (deep learning)
    reports/08_transformers/comparacion_transformers.csv (transformers)

Salidas (reports/09_comparacion/):
    comparacion_global.csv                 (todas las filas, todas las familias)
    comparacion_global_test.csv            (solo prueba, ordenado por F1-macro)
    comparacion_global_f1_macro.png        (grafico de barras, F1-macro en test)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS = PROJECT_ROOT / "reports"
SALIDA_DIR = REPORTS / "09_comparacion"

# (etiqueta legible, ruta del CSV de cada fase)
FUENTES = [
    ("clasicos (fase 07)", REPORTS / "06_modelado" / "comparacion_modelos.csv"),
    ("deep learning (fase 08)", REPORTS / "07_dl" / "comparacion_dl.csv"),
    ("transformers (fase 09)", REPORTS / "08_transformers" / "comparacion_transformers.csv"),
]

# Color por familia para el grafico.
COLORES = {
    "clasico": "#4C72B0",
    "deep_learning": "#DD8452",
    "transformer": "#55A868",
}


def cargar_fuentes():
    marcos, faltantes = [], []
    for etiqueta, ruta in FUENTES:
        if ruta.exists():
            marcos.append(pd.read_csv(ruta))
            print(f"  OK  {etiqueta}: {ruta.relative_to(PROJECT_ROOT)}")
        else:
            faltantes.append(etiqueta)
            print(f"  --  {etiqueta}: falta {ruta.relative_to(PROJECT_ROOT)} (corre esa fase primero)")
    return marcos, faltantes


def graficar(tabla_test):
    tabla_test = tabla_test.sort_values("f1_macro", ascending=True)
    etiquetas = [f"{m}\n{e}" for m, e in zip(tabla_test["modelo"], tabla_test["estrategia"])]
    colores = [COLORES.get(f, "#888888") for f in tabla_test["familia"]]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(tabla_test))))
    barras = ax.barh(etiquetas, tabla_test["f1_macro"], color=colores)
    ax.set_xlabel("F1-Macro (prueba)")
    ax.set_title("Comparacion de modelos por familia — F1-Macro en prueba")
    ax.set_xlim(0, max(0.7, float(tabla_test["f1_macro"].max()) + 0.05))
    for barra, valor in zip(barras, tabla_test["f1_macro"]):
        ax.text(valor + 0.005, barra.get_y() + barra.get_height() / 2, f"{valor:.3f}", va="center", fontsize=9)

    # Leyenda por familia.
    presentes = list(dict.fromkeys(tabla_test["familia"]))
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORES.get(f, "#888888")) for f in presentes]
    ax.legend(handles, presentes, title="familia", loc="lower right")

    fig.tight_layout()
    fig.savefig(SALIDA_DIR / "comparacion_global_f1_macro.png", dpi=120)
    plt.close(fig)


def main():
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    print("Reuniendo comparaciones de cada fase:")
    marcos, faltantes = cargar_fuentes()

    if not marcos:
        print("\nNo hay ninguna comparacion para unir. Corre primero las fases 07, 08 y/o 09.")
        sys.exit(1)

    global_df = pd.concat(marcos, ignore_index=True)
    global_df.to_csv(SALIDA_DIR / "comparacion_global.csv", index=False, encoding="utf-8-sig")

    tabla_test = global_df[global_df["split"] == "test"].sort_values("f1_macro", ascending=False).reset_index(drop=True)
    tabla_test.to_csv(SALIDA_DIR / "comparacion_global_test.csv", index=False, encoding="utf-8-sig")
    graficar(tabla_test)

    print("\n" + "=" * 86)
    print("COMPARACION GLOBAL - todas las familias (F1-macro en prueba, ordenado)")
    print("=" * 86)
    print(tabla_test.to_string(index=False))

    print("\nMejor combinacion por familia (F1-macro en prueba):")
    for familia, grupo in tabla_test.groupby("familia"):
        fila = grupo.iloc[0]
        print(f"  {familia:<14} -> {fila['modelo']} ({fila['estrategia']}): {fila['f1_macro']:.4f}")

    mejor = tabla_test.iloc[0]
    print(f"\nMejor modelo global: {mejor['familia']} / {mejor['modelo']} ({mejor['estrategia']}) "
          f"con F1-macro {mejor['f1_macro']:.4f} en prueba")
    if faltantes:
        print(f"\nNota: faltan familias por correr: {', '.join(faltantes)}")
    print(f"\nSalidas en: {SALIDA_DIR}")


if __name__ == "__main__":
    main()
