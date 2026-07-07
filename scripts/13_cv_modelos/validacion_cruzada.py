"""Fase 13: validacion cruzada estratificada + hiperparametros sobre la semilla de 500.

Pasos 5-7 del flujo del docente: con los 500 registros etiquetados (fase 12) se
comparan los SEIS modelos del proyecto mediante validacion cruzada estratificada de
5 particiones, explorando una malla de hiperparametros por algoritmo:

    svm          -> C (regularizacion), class_weight balanced fijo
    naive_bayes  -> alpha (suavizado) x uso de SMOTE
    cnn          -> learning rate x numero de filtros
    lstm         -> learning rate x tamano oculto
    beto         -> learning rate (fine-tuning)
    xlm_roberta  -> learning rate (fine-tuning)

Para cada configuracion se reporta el F1-macro promedio entre folds (con desviacion
estandar) y las metricas agregadas (pooled) sobre las predicciones out-of-fold. La
mejor configuracion por modelo y el mejor modelo global (por F1-macro promedio) se
guardan en mejores_hiperparametros.json, que consumen las fases 14 y 15.

Si el equipo corrigio etiquetas en reports/11_etiquetado_llm/revision_equipo_semilla.csv
(columna etiqueta_corregida), correr con --aplicar-correcciones para usarlas.

Entrada:
    data/splits_v2/semilla_etiquetada.csv

Salidas (reports/12_cv_modelos/):
    cv_resultados.csv                (todas las configuraciones)
    cv_mejor_por_modelo.csv
    mejores_hiperparametros.json
    matriz_confusion_cv_<modelo>.csv/.png   (mejor configuracion, out-of-fold)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
from _comun.entrenadores import (  # noqa: E402
    COLUMNA_TEXTO, FAMILIAS, dispositivo, entrenar_modelo, predecir_modelo,
)
from _comun.evaluacion import CLASES, metricas_basicas  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "data" / "splits_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "12_cv_modelos"
REVISION_FILE = PROJECT_ROOT / "reports" / "11_etiquetado_llm" / "revision_equipo_semilla.csv"

# Mallas de hiperparametros por modelo (acotadas para que la CV sea tratable).
GRIDS = {
    "svm": [{"C": 0.5}, {"C": 1.0}, {"C": 5.0}],
    "naive_bayes": [{"alpha": a, "usar_smote": s} for a in (0.1, 0.5, 1.0) for s in (False, True)],
    "cnn": [{"lr": lr, "num_filtros": f} for lr in (1e-3, 5e-4) for f in (100, 150)],
    "lstm": [{"lr": lr, "hidden": h} for lr in (1e-3, 5e-4) for h in (128, 256)],
    "beto": [{"lr": 2e-5}, {"lr": 3e-5}],
    "xlm_roberta": [{"lr": 2e-5}, {"lr": 3e-5}],
}


def cargar_semilla(aplicar_correcciones):
    df = pd.read_csv(SPLITS_DIR / "semilla_etiquetada.csv").fillna("")
    if aplicar_correcciones and REVISION_FILE.exists():
        rev = pd.read_csv(REVISION_FILE).fillna("")
        rev = rev[rev["etiqueta_corregida"].isin(CLASES)][["id_registro", "etiqueta_corregida"]]
        if len(rev):
            df = df.merge(rev, on="id_registro", how="left").fillna("")
            corregidas = df["etiqueta_corregida"].isin(CLASES)
            df.loc[corregidas, "sentimiento_v2"] = df.loc[corregidas, "etiqueta_corregida"]
            print(f"Correcciones del equipo aplicadas: {int(corregidas.sum())}")
    df = df[df["sentimiento_v2"].isin(CLASES)].reset_index(drop=True)
    return df


def guardar_matriz_cv(y_true, y_pred, modelo):
    matriz = confusion_matrix(y_true, y_pred, labels=CLASES)
    pd.DataFrame(matriz, index=CLASES, columns=CLASES).to_csv(
        REPORT_DIR / f"matriz_confusion_cv_{modelo}.csv", encoding="utf-8-sig")
    display = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=CLASES)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"CV out-of-fold - {modelo}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"matriz_confusion_cv_{modelo}.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CV 5-fold + hiperparametros sobre la semilla (fase 13).")
    parser.add_argument("--modelos", nargs="+", default=list(GRIDS), choices=list(GRIDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--aplicar-correcciones", action="store_true",
                        help="Usa etiqueta_corregida del CSV de revision del equipo.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = dispositivo(args.cpu)
    print(f"Dispositivo: {device}")

    df = cargar_semilla(args.aplicar_correcciones)
    print(f"Semilla utilizable: {len(df)} filas")
    print(df["sentimiento_v2"].value_counts().to_string())

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    y = df["sentimiento_v2"]

    filas, predicciones_oof = [], {}
    for modelo in args.modelos:
        columna = COLUMNA_TEXTO[FAMILIAS[modelo]]
        textos = df[columna].astype(str)
        for config in GRIDS[modelo]:
            etiqueta_config = json.dumps(config, sort_keys=True)
            inicio = time.time()
            f1_folds, y_true_pool, y_pred_pool = [], [], []
            for fold, (idx_tr, idx_va) in enumerate(skf.split(textos, y), start=1):
                art = entrenar_modelo(modelo, config,
                                      textos.iloc[idx_tr], y.iloc[idx_tr],
                                      textos.iloc[idx_va], y.iloc[idx_va],
                                      device, random_state=args.random_state)
                pred, _, _ = predecir_modelo(art, textos.iloc[idx_va], device)
                m = metricas_basicas(y.iloc[idx_va].tolist(), pred)
                f1_folds.append(m["f1_macro"])
                y_true_pool.extend(y.iloc[idx_va].tolist())
                y_pred_pool.extend(pred)
                del art
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            agregadas = metricas_basicas(y_true_pool, y_pred_pool)
            fila = {
                "familia": FAMILIAS[modelo], "modelo": modelo, "config": etiqueta_config,
                "f1_macro_medio": round(float(np.mean(f1_folds)), 4),
                "f1_macro_std": round(float(np.std(f1_folds)), 4),
                **{f"{k}_oof": v for k, v in agregadas.items()},
                "segundos": round(time.time() - inicio, 1),
            }
            filas.append(fila)
            predicciones_oof[(modelo, etiqueta_config)] = (y_true_pool, y_pred_pool)
            print(f">> {modelo:<12} {etiqueta_config:<42} | F1-macro CV "
                  f"{fila['f1_macro_medio']:.4f} +/- {fila['f1_macro_std']:.4f} "
                  f"| {fila['segundos']}s", flush=True)

    resultados = pd.DataFrame(filas).sort_values("f1_macro_medio", ascending=False)
    resultados.to_csv(REPORT_DIR / "cv_resultados.csv", index=False, encoding="utf-8-sig")

    mejores = resultados.loc[resultados.groupby("modelo")["f1_macro_medio"].idxmax()]
    mejores = mejores.sort_values("f1_macro_medio", ascending=False)
    mejores.to_csv(REPORT_DIR / "cv_mejor_por_modelo.csv", index=False, encoding="utf-8-sig")

    for _, fila in mejores.iterrows():
        y_true_pool, y_pred_pool = predicciones_oof[(fila["modelo"], fila["config"])]
        guardar_matriz_cv(y_true_pool, y_pred_pool, fila["modelo"])

    mejor_global = mejores.iloc[0]
    salida_json = {
        "mejor_global": {
            "modelo": mejor_global["modelo"],
            "familia": mejor_global["familia"],
            "config": json.loads(mejor_global["config"]),
            "f1_macro_medio": float(mejor_global["f1_macro_medio"]),
        },
        "por_modelo": {
            fila["modelo"]: {
                "familia": fila["familia"],
                "config": json.loads(fila["config"]),
                "f1_macro_medio": float(fila["f1_macro_medio"]),
                "f1_macro_std": float(fila["f1_macro_std"]),
            }
            for _, fila in mejores.iterrows()
        },
        "folds": args.folds,
        "n_semilla": len(df),
        "random_state": args.random_state,
    }
    with open(REPORT_DIR / "mejores_hiperparametros.json", "w", encoding="utf-8") as f:
        json.dump(salida_json, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print("MEJOR CONFIGURACION POR MODELO (F1-macro promedio de CV)")
    print("=" * 78)
    print(mejores[["familia", "modelo", "config", "f1_macro_medio", "f1_macro_std",
                   "balanced_accuracy_oof", "accuracy_oof"]].to_string(index=False))
    print(f"\nMejor modelo global: {mejor_global['modelo']} "
          f"(F1-macro CV {mejor_global['f1_macro_medio']:.4f})")
    print(f"Hiperparametros guardados en: {REPORT_DIR / 'mejores_hiperparametros.json'}")


if __name__ == "__main__":
    main()
