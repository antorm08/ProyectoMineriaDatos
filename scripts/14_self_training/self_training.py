"""Fase 14: self-training del mejor modelo + verificacion por clustering + revision.

Pasos 8-12 del flujo del docente:

    8.  Reentrenar el mejor modelo de la CV (fase 13) con los 500 de la semilla.
    9.  Usarlo para etiquetar automaticamente el resto del 80% (dev_resto).
    10. Aplicar clustering (TF-IDF + SVD + K-Means) sobre esos textos y comparar la
        etiqueta asignada con la etiqueta mayoritaria de cada cluster.
    11. Exportar los casos dudosos para revision: baja confianza del modelo o
        discrepancia con un cluster suficientemente puro. Si existe un archivo de
        revision con correcciones, se integran antes de consolidar el dataset.
    12. Consolidar el dataset de entrenamiento completo etiquetado.

El reentrenamiento del paso 8 reserva internamente un 15% de la semilla para el
early stopping de redes/transformers (los clasicos usan los 500 completos).

Entradas:
    data/splits_v2/semilla_etiquetada.csv
    data/splits_v2/dev_resto.csv
    reports/12_cv_modelos/mejores_hiperparametros.json

Salidas:
    data/splits_v2/dev_etiquetado_completo.csv     (80% etiquetado, con origen)
    reports/13_self_training/resumen_self_training.csv
    reports/13_self_training/pureza_clusters.csv
    reports/13_self_training/casos_dudosos_revision.csv
    reports/13_self_training/cambios_revision.csv
    reports/13_self_training/distribucion_final_dev.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
from _comun.entrenadores import (  # noqa: E402
    COLUMNA_TEXTO, construir_vectorizador, dispositivo, entrenar_modelo, predecir_modelo,
)
from _comun.evaluacion import CLASES  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "data" / "splits_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "13_self_training"
CV_JSON = PROJECT_ROOT / "reports" / "12_cv_modelos" / "mejores_hiperparametros.json"
REVISION_SEMILLA_FILE = PROJECT_ROOT / "reports" / "11_etiquetado_manual" / "revision_semilla_500.csv"
REVISION_DUDOSOS_FILE = REPORT_DIR / "casos_dudosos_revision.csv"


def cargar_semilla(aplicar_correcciones):
    df = pd.read_csv(SPLITS_DIR / "semilla_etiquetada.csv").fillna("")
    origen = pd.Series("manual_semilla_500", index=df.index)
    if aplicar_correcciones and REVISION_SEMILLA_FILE.exists():
        rev = pd.read_csv(REVISION_SEMILLA_FILE).fillna("")
        rev = rev[rev["etiqueta_corregida"].isin(CLASES)][["id_registro", "etiqueta_corregida"]]
        if len(rev):
            df = df.merge(rev, on="id_registro", how="left").fillna("")
            corregidas = df["etiqueta_corregida"].isin(CLASES)
            df.loc[corregidas, "sentimiento_v2"] = df.loc[corregidas, "etiqueta_corregida"]
            origen[corregidas.values] = "equipo_correccion"
            print(f"Correcciones del equipo aplicadas: {int(corregidas.sum())}")
    df["origen_etiqueta_v2"] = origen.values
    return df[df["sentimiento_v2"].isin(CLASES)].reset_index(drop=True)


def clustering_coherencia(dev, umbral_pureza, k, random_state):
    """K-Means sobre TF-IDF+SVD; marca filas que discrepan de clusters puros."""
    vectorizador = construir_vectorizador(20000, usar_stopwords=True)
    x = vectorizador.fit_transform(dev["texto_modelo"].astype(str))
    svd = TruncatedSVD(n_components=100, random_state=random_state)
    emb = svd.fit_transform(x)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    dev = dev.copy()
    dev["cluster"] = kmeans.fit_predict(emb)

    filas_pureza, discrepantes = [], pd.Series(False, index=dev.index)
    for c, grupo in dev.groupby("cluster"):
        conteo = grupo["etiqueta_modelo"].value_counts()
        mayoritaria = conteo.idxmax()
        pureza = conteo.max() / len(grupo)
        filas_pureza.append({
            "cluster": c, "tamano": len(grupo), "etiqueta_mayoritaria": mayoritaria,
            "pureza": round(float(pureza), 4),
        })
        if pureza >= umbral_pureza:
            discrepantes |= (dev["cluster"] == c) & (dev["etiqueta_modelo"] != mayoritaria)
    return dev, pd.DataFrame(filas_pureza).sort_values("pureza"), discrepantes


def exportar_y_aplicar_revision(dev, dudosos):
    """Exporta dudosos y aplica correcciones manuales si el CSV ya las contiene."""
    columnas = [
        "id_registro", "empresa", "rubro", "estrellas", "comentario_limpio",
        "etiqueta_modelo", "confianza_modelo", "cluster", "sentimiento_final",
    ]
    revision = dudosos[[c for c in columnas if c in dudosos.columns]].copy()
    if "etiqueta_revisada" not in revision.columns:
        revision["etiqueta_revisada"] = ""
    if "observacion_revision" not in revision.columns:
        revision["observacion_revision"] = ""

    if REVISION_DUDOSOS_FILE.exists():
        existente = pd.read_csv(REVISION_DUDOSOS_FILE).fillna("")
        if "etiqueta_revisada" in existente.columns:
            correcciones = existente[existente["etiqueta_revisada"].isin(CLASES)][
                ["id_registro", "etiqueta_revisada"]
            ]
            if len(correcciones):
                revision = revision.drop(columns=["etiqueta_revisada"], errors="ignore")
                revision = revision.merge(correcciones, on="id_registro", how="left").fillna("")

    revision.to_csv(REVISION_DUDOSOS_FILE, index=False, encoding="utf-8-sig")

    cambios = []
    for fila in revision[revision["etiqueta_revisada"].isin(CLASES)].itertuples():
        idx = dev.index[dev["id_registro"] == fila.id_registro]
        if len(idx):
            i = idx[0]
            if fila.etiqueta_revisada != dev.loc[i, "etiqueta_modelo"]:
                cambios.append({
                    "id_registro": fila.id_registro,
                    "etiqueta_modelo": dev.loc[i, "etiqueta_modelo"],
                    "etiqueta_revisada": fila.etiqueta_revisada,
                    "confianza_modelo": dev.loc[i, "confianza_modelo"],
                })
            dev.loc[i, "sentimiento_v2"] = fila.etiqueta_revisada
            dev.loc[i, "origen_etiqueta_v2"] = "revision_manual"
    return dev, pd.DataFrame(cambios)


def main():
    parser = argparse.ArgumentParser(description="Self-training + clustering + revision manual (fase 14).")
    parser.add_argument("--umbral-confianza", type=float, default=0.50)
    parser.add_argument("--umbral-pureza", type=float, default=0.60)
    parser.add_argument("--clusters", type=int, default=25)
    parser.add_argument("--max-revision", type=int, default=900,
                        help="Tope de casos exportados para revision (menor confianza primero).")
    parser.add_argument("--aplicar-correcciones", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    device = dispositivo(args.cpu)
    print(f"Dispositivo: {device}")

    with open(CV_JSON, encoding="utf-8") as f:
        cv = json.load(f)
    nombre = cv["mejor_global"]["modelo"]
    config = cv["mejor_global"]["config"]
    columna = COLUMNA_TEXTO[cv["mejor_global"]["familia"]]
    print(f"Mejor modelo de la CV: {nombre} | config {config}")

    semilla = cargar_semilla(args.aplicar_correcciones)
    dev = pd.read_csv(SPLITS_DIR / "dev_resto.csv").fillna("")

    # Paso 8: reentrenar el mejor modelo con la semilla (15% interno para early stopping).
    ent, ret = train_test_split(semilla, test_size=0.15, random_state=args.random_state,
                                stratify=semilla["sentimiento_v2"])
    print(f"Reentrenando {nombre} con semilla: {len(ent)} train / {len(ret)} retencion")
    artefacto = entrenar_modelo(nombre, config, ent[columna].astype(str), ent["sentimiento_v2"],
                                ret[columna].astype(str), ret["sentimiento_v2"],
                                device, random_state=args.random_state, verbose=True)

    # Paso 9: etiquetar automaticamente el resto del 80%.
    print(f"Etiquetando dev_resto: {len(dev)} filas")
    pred, scores, orden = predecir_modelo(artefacto, dev[columna].astype(str), device)
    scores = np.asarray(scores)
    dev["etiqueta_modelo"] = pred
    dev["confianza_modelo"] = scores.max(axis=1).round(4)
    dev["sentimiento_v2"] = dev["etiqueta_modelo"]
    dev["origen_etiqueta_v2"] = "modelo_self_training"

    # Paso 10: clustering de coherencia.
    dev, pureza, discrepantes = clustering_coherencia(dev, args.umbral_pureza,
                                                      args.clusters, args.random_state)
    baja_confianza = dev["confianza_modelo"] < args.umbral_confianza
    dudosos = dev[baja_confianza | discrepantes].sort_values("confianza_modelo")
    if len(dudosos) > args.max_revision:
        dudosos = dudosos.head(args.max_revision)
    print(f"Dudosos: {int(baja_confianza.sum())} por confianza, {int(discrepantes.sum())} "
          f"por clustering -> {len(dudosos)} a revision (tope {args.max_revision})")

    # Paso 11: exportar dudosos y aplicar correcciones manuales si existen.
    dev, cambios = exportar_y_aplicar_revision(dev, dudosos)

    # Paso 12: consolidar el 80% etiquetado.
    columnas_extra = ["etiqueta_modelo", "confianza_modelo", "cluster"]
    semilla_out = semilla.copy()
    for col in columnas_extra:
        semilla_out[col] = ""
    completo = pd.concat([semilla_out, dev], ignore_index=True)
    completo.to_csv(SPLITS_DIR / "dev_etiquetado_completo.csv", index=False, encoding="utf-8-sig")

    pureza.to_csv(REPORT_DIR / "pureza_clusters.csv", index=False, encoding="utf-8-sig")
    cambios.to_csv(REPORT_DIR / "cambios_revision.csv", index=False, encoding="utf-8-sig")
    dist = completo["sentimiento_v2"].value_counts().rename_axis("clase").reset_index(name="cantidad")
    dist.to_csv(REPORT_DIR / "distribucion_final_dev.csv", index=False, encoding="utf-8-sig")

    resumen = pd.DataFrame([
        ("modelo_self_training", nombre),
        ("config", json.dumps(config)),
        ("filas_semilla", len(semilla)),
        ("filas_dev_resto", len(dev)),
        ("dudosos_confianza", int(baja_confianza.sum())),
        ("dudosos_clustering", int(discrepantes.sum())),
        ("enviados_revision", len(dudosos)),
        ("cambiados_por_revision", len(cambios)),
        ("umbral_confianza", args.umbral_confianza),
        ("umbral_pureza", args.umbral_pureza),
        ("clusters", args.clusters),
        ("filas_dev_completo", len(completo)),
    ], columns=["metrica", "valor"])
    resumen.to_csv(REPORT_DIR / "resumen_self_training.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print(resumen.to_string(index=False))
    print("\nDistribucion final del 80%:")
    print(dist.to_string(index=False))
    print(f"\nDataset consolidado: {SPLITS_DIR / 'dev_etiquetado_completo.csv'}")


if __name__ == "__main__":
    main()
