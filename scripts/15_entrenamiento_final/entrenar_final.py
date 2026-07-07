"""Fase 15: entrenamiento final de los 6 modelos, curvas y evaluacion unica en test.

Pasos 13-16 del flujo del docente:

    13. Entrenar nuevamente los 6 modelos con el 80% etiquetado (fase 14), usando la
        mejor configuracion de hiperparametros hallada en la CV (fase 13). Dentro del
        80% se aplica la subdivision 70/30 entrenamiento/validacion pedida en la
        retroalimentacion ("80 (70-30) - 20").
    14. Comparar los 6 modelos finales sobre validacion.
    15. Elegir el mejor modelo final por F1-macro en validacion.
    16. Evaluar UNA SOLA VEZ ese mejor modelo en el 20% de prueba reservado en la
        fase 11 (etiquetas de referencia: protocolo LLM de la fase 12).

Solo el modelo ganador toca el conjunto de prueba: asi la estimacion final no sufre
sesgo de seleccion sobre el test. Ademas se grafican las curvas de entrenamiento
(perdida y F1-macro por epoca) de redes y transformers, y curvas de aprendizaje
(F1 vs tamano de entrenamiento) de los clasicos.

Entradas:
    data/splits_v2/dev_etiquetado_completo.csv
    data/splits_v2/test_etiquetado.csv
    reports/12_cv_modelos/mejores_hiperparametros.json

Salidas (reports/14_entrenamiento_final/):
    comparacion_valid.csv, f1_por_clase_valid.csv
    curvas_entrenamiento.csv + curva_entrenamiento_<modelo>.png
    curva_aprendizaje_<clasico>.png
    matriz_confusion / reporte / ROC / predicciones del mejor en test
    resumen_final.csv, comparacion_final_f1.png
models/:
    v2_mejor_modelo.joblib | v2_mejor_modelo_dl.pt | v2_mejor_modelo_transformer/
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
from _comun.entrenadores import (  # noqa: E402
    COLUMNA_TEXTO, FAMILIAS, dispositivo, entrenar_modelo, predecir_modelo,
)
from _comun.evaluacion import CLASES, evaluar_split, metricas_basicas  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "data" / "splits_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "14_entrenamiento_final"
MODELS_DIR = PROJECT_ROOT / "models"
CV_JSON = PROJECT_ROOT / "reports" / "12_cv_modelos" / "mejores_hiperparametros.json"


def graficar_curva_entrenamiento(historial, modelo):
    epocas = [h["epoca"] for h in historial]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(epocas, [h["loss_train"] for h in historial], "o-", color="tab:red")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Perdida de entrenamiento", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(epocas, [h["f1_valid"] for h in historial], "s-", color="tab:blue")
    ax2.set_ylabel("F1-macro validacion", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title(f"Curva de entrenamiento - {modelo}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"curva_entrenamiento_{modelo}.png", dpi=120)
    plt.close(fig)


def curva_aprendizaje_clasico(nombre, config, train, valid, columna, device, random_state):
    """F1-macro en validacion vs fraccion del entrenamiento (curva de aprendizaje)."""
    fracciones = [0.2, 0.4, 0.6, 0.8, 1.0]
    puntos = []
    for frac in fracciones:
        if frac < 1.0:
            sub, _ = train_test_split(train, train_size=frac, random_state=random_state,
                                      stratify=train["sentimiento_v2"])
        else:
            sub = train
        art = entrenar_modelo(nombre, config, sub[columna].astype(str), sub["sentimiento_v2"],
                              valid[columna].astype(str), valid["sentimiento_v2"],
                              device, random_state=random_state)
        pred, _, _ = predecir_modelo(art, valid[columna].astype(str), device)
        m = metricas_basicas(valid["sentimiento_v2"].tolist(), pred)
        puntos.append({"fraccion": frac, "n_train": len(sub), "f1_macro": m["f1_macro"]})

    df = pd.DataFrame(puntos)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["n_train"], df["f1_macro"], "o-")
    ax.set_xlabel("Ejemplos de entrenamiento")
    ax.set_ylabel("F1-macro validacion")
    ax.set_title(f"Curva de aprendizaje - {nombre}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"curva_aprendizaje_{nombre}.png", dpi=120)
    plt.close(fig)
    return df


def guardar_mejor(artefacto):
    familia = artefacto["familia"]
    if familia == "clasico":
        ruta = MODELS_DIR / "v2_mejor_modelo.joblib"
        joblib.dump({
            "vectorizador": artefacto["vectorizador"],
            "clasificador": artefacto["clasificador"],
            "clases": CLASES, "algoritmo": artefacto["nombre"], "config": artefacto["config"],
        }, ruta)
    elif familia == "deep_learning":
        ruta = MODELS_DIR / "v2_mejor_modelo_dl.pt"
        torch.save({
            "state_dict": artefacto["modelo"].state_dict(),
            "arquitectura": artefacto["nombre"], "config": artefacto["config"],
            "vocab": artefacto["vocab"], "clases": CLASES,
        }, ruta)
    else:
        ruta = MODELS_DIR / "v2_mejor_modelo_transformer"
        ruta.mkdir(parents=True, exist_ok=True)
        artefacto["modelo"].save_pretrained(ruta)
        artefacto["tokenizer"].save_pretrained(ruta)
    return ruta


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento final y evaluacion unica en test (fase 15).")
    parser.add_argument("--modelos", nargs="+", default=list(FAMILIAS), choices=list(FAMILIAS))
    parser.add_argument("--valid-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    device = dispositivo(args.cpu)
    print(f"Dispositivo: {device}")

    with open(CV_JSON, encoding="utf-8") as f:
        cv = json.load(f)

    dev = pd.read_csv(SPLITS_DIR / "dev_etiquetado_completo.csv").fillna("")
    dev = dev[dev["sentimiento_v2"].isin(CLASES)]
    test = pd.read_csv(SPLITS_DIR / "test_etiquetado.csv").fillna("")
    test = test[test["sentimiento_v2"].isin(CLASES)]

    train, valid = train_test_split(dev, test_size=args.valid_size,
                                    random_state=args.random_state,
                                    stratify=dev["sentimiento_v2"])
    print(f"Datos: {len(train)} train / {len(valid)} valid (70/30 del 80%) | test {len(test)}")

    filas_comparacion, filas_f1_clase, curvas, artefactos = [], [], [], {}
    for nombre in args.modelos:
        config = cv["por_modelo"][nombre]["config"]
        columna = COLUMNA_TEXTO[FAMILIAS[nombre]]
        print(f"\n>> Entrenamiento final: {nombre} | config {config}")
        artefacto = entrenar_modelo(nombre, config,
                                    train[columna].astype(str), train["sentimiento_v2"],
                                    valid[columna].astype(str), valid["sentimiento_v2"],
                                    device, random_state=args.random_state, verbose=True)
        pred_valid, _, _ = predecir_modelo(artefacto, valid[columna].astype(str), device)
        fila = evaluar_split(valid["sentimiento_v2"].tolist(), pred_valid,
                             FAMILIAS[nombre], nombre, "final", "valid",
                             REPORT_DIR, filas_comparacion, filas_f1_clase)
        print(f"   -> valid F1-macro {fila['f1_macro']:.4f}")

        if artefacto["historial"]:
            graficar_curva_entrenamiento(artefacto["historial"], nombre)
            for h in artefacto["historial"]:
                curvas.append({"modelo": nombre, **h})
        else:
            curva = curva_aprendizaje_clasico(nombre, config, train, valid, columna,
                                              device, args.random_state)
            for _, p in curva.iterrows():
                curvas.append({"modelo": nombre, "epoca": None,
                               "n_train": int(p["n_train"]), "f1_valid": p["f1_macro"]})
        artefactos[nombre] = {"artefacto": artefacto, "valid": fila}
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparacion = pd.DataFrame(filas_comparacion)
    comparacion.to_csv(REPORT_DIR / "comparacion_valid.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_f1_clase).to_csv(REPORT_DIR / "f1_por_clase_valid.csv",
                                        index=False, encoding="utf-8-sig")
    pd.DataFrame(curvas).to_csv(REPORT_DIR / "curvas_entrenamiento.csv",
                                index=False, encoding="utf-8-sig")

    # Paso 15: mejor modelo final por F1-macro en validacion.
    mejor = max(artefactos, key=lambda n: artefactos[n]["valid"]["f1_macro"])
    artefacto = artefactos[mejor]["artefacto"]
    print(f"\nMejor modelo final: {mejor} (valid F1-macro "
          f"{artefactos[mejor]['valid']['f1_macro']:.4f})")

    # Paso 16: evaluacion UNICA en el 20% de prueba (solo el mejor).
    columna = COLUMNA_TEXTO[FAMILIAS[mejor]]
    pred_test, scores_test, orden = predecir_modelo(artefacto, test[columna].astype(str), device)
    fila_test = evaluar_split(test["sentimiento_v2"].tolist(), pred_test,
                              FAMILIAS[mejor], mejor, "final", "test",
                              REPORT_DIR, filas_comparacion, filas_f1_clase,
                              scores=scores_test, score_labels=orden)
    ruta_modelo = guardar_mejor(artefacto)

    # Grafico comparativo (valid) + fila de test del mejor.
    tabla_valid = comparacion[comparacion["split"] == "valid"].sort_values("f1_macro")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colores = {"clasico": "tab:gray", "deep_learning": "tab:orange", "transformer": "tab:blue"}
    ax.barh(tabla_valid["modelo"], tabla_valid["f1_macro"],
            color=[colores[f] for f in tabla_valid["familia"]])
    ax.set_xlabel("F1-macro validacion")
    ax.set_title("Comparacion final de los 6 modelos (validacion 30%)")
    for i, v in enumerate(tabla_valid["f1_macro"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "comparacion_final_f1.png", dpi=120)
    plt.close(fig)

    resumen = pd.DataFrame([
        ("mejor_modelo", mejor),
        ("familia", FAMILIAS[mejor]),
        ("config", json.dumps(cv["por_modelo"][mejor]["config"])),
        ("f1_macro_valid", artefactos[mejor]["valid"]["f1_macro"]),
        ("f1_macro_test", fila_test["f1_macro"]),
        ("balanced_accuracy_test", fila_test["balanced_accuracy"]),
        ("f1_weighted_test", fila_test["f1_weighted"]),
        ("accuracy_test", fila_test["accuracy"]),
        ("n_train", len(train)), ("n_valid", len(valid)), ("n_test", len(test)),
        ("modelo_guardado", str(ruta_modelo)),
    ], columns=["metrica", "valor"])
    resumen.to_csv(REPORT_DIR / "resumen_final.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("COMPARACION FINAL EN VALIDACION")
    print("=" * 78)
    print(comparacion[comparacion["split"] == "valid"]
          .sort_values("f1_macro", ascending=False).to_string(index=False))
    print("\nEVALUACION UNICA EN TEST (solo el mejor)")
    print(pd.DataFrame([fila_test]).to_string(index=False))
    print(f"\nModelo guardado en: {ruta_modelo}")
    print(f"Reportes en: {REPORT_DIR}")


if __name__ == "__main__":
    main()
