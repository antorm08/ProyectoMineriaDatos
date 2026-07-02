"""Fase 14: self-training del mejor modelo + verificacion por clustering + arbitraje.

Pasos 8-12 del flujo del docente:

    8.  Reentrenar el mejor modelo de la CV (fase 13) con los 500 de la semilla.
    9.  Usarlo para etiquetar automaticamente el resto del 80% (dev_resto).
    10. Aplicar clustering (TF-IDF + SVD + K-Means) sobre esos textos y comparar la
        etiqueta asignada con la etiqueta mayoritaria de cada cluster.
    11. Revisar los casos dudosos: baja confianza del modelo o discrepancia con un
        cluster suficientemente puro. El arbitraje lo hace el LLM anotador (mismo
        protocolo de la fase 12), que actua como revisor de segunda opinion.
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
    reports/13_self_training/cambios_arbitraje.csv
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
from _comun.llm import etiquetar_lote  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "data" / "splits_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "13_self_training"
CV_JSON = PROJECT_ROOT / "reports" / "12_cv_modelos" / "mejores_hiperparametros.json"
REVISION_FILE = PROJECT_ROOT / "reports" / "11_etiquetado_llm" / "revision_equipo_semilla.csv"
CACHE_ARBITRAJE = SPLITS_DIR / "cache_arbitraje_llm.csv"


def cargar_semilla(aplicar_correcciones):
    df = pd.read_csv(SPLITS_DIR / "semilla_etiquetada.csv").fillna("")
    origen = pd.Series("llm_semilla", index=df.index)
    if aplicar_correcciones and REVISION_FILE.exists():
        rev = pd.read_csv(REVISION_FILE).fillna("")
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


def arbitrar_con_llm(dev, indices, modelo_llm, proveedor, batch):
    """Pide al LLM segunda opinion para las filas dudosas. Cachea el avance."""
    cache = {}
    if CACHE_ARBITRAJE.exists():
        for fila in pd.read_csv(CACHE_ARBITRAJE).itertuples():
            cache[fila.id_registro] = fila

    pendientes = [i for i in indices if dev.loc[i, "id_registro"] not in cache]
    print(f"Arbitraje LLM: {len(indices)} dudosos ({len(pendientes)} pendientes de consulta)")

    for i in range(0, len(pendientes), batch):
        lote_idx = pendientes[i:i + batch]
        ids = dev.loc[lote_idx, "id_registro"].tolist()
        textos = dev.loc[lote_idx, "comentario_limpio"].tolist()
        resultados = etiquetar_lote(textos, modelo=modelo_llm, proveedor=proveedor, ids=ids)
        filas = []
        for id_reg, res in zip(ids, resultados):
            filas.append({
                "id_registro": id_reg,
                "etiqueta_llm": res["etiqueta"] if res else "",
                "confianza_llm": res["confianza"] if res else 0.0,
                "justificacion_llm": res["justificacion"] if res else "sin respuesta valida",
            })
        pd.DataFrame(filas).to_csv(CACHE_ARBITRAJE, mode="a",
                                   header=not CACHE_ARBITRAJE.exists(),
                                   index=False, encoding="utf-8-sig")
        for fila in filas:
            cache[fila["id_registro"]] = pd.Series(fila)
        print(f"   {min(i + batch, len(pendientes))}/{len(pendientes)} arbitradas", flush=True)

    cambios = []
    for i in indices:
        id_reg = dev.loc[i, "id_registro"]
        if id_reg not in cache:
            continue
        registro = cache[id_reg]
        etiqueta_llm = getattr(registro, "etiqueta_llm", None) or registro["etiqueta_llm"]
        if etiqueta_llm in CLASES:
            if etiqueta_llm != dev.loc[i, "etiqueta_modelo"]:
                cambios.append({
                    "id_registro": id_reg,
                    "etiqueta_modelo": dev.loc[i, "etiqueta_modelo"],
                    "etiqueta_arbitraje": etiqueta_llm,
                    "confianza_modelo": dev.loc[i, "confianza_modelo"],
                })
            dev.loc[i, "sentimiento_v2"] = etiqueta_llm
            dev.loc[i, "origen_etiqueta_v2"] = "llm_arbitraje"
    return dev, pd.DataFrame(cambios)


def main():
    parser = argparse.ArgumentParser(description="Self-training + clustering + arbitraje (fase 14).")
    parser.add_argument("--modelo-etiquetador", required=True, help="Id del LLM arbitro.")
    parser.add_argument("--proveedor", default="nvidia", choices=["nvidia", "openrouter"])
    parser.add_argument("--umbral-confianza", type=float, default=0.50)
    parser.add_argument("--umbral-pureza", type=float, default=0.60)
    parser.add_argument("--clusters", type=int, default=25)
    parser.add_argument("--batch", type=int, default=15)
    parser.add_argument("--max-arbitraje", type=int, default=900,
                        help="Tope de casos enviados al LLM (los de menor confianza primero).")
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
    if len(dudosos) > args.max_arbitraje:
        dudosos = dudosos.head(args.max_arbitraje)
    print(f"Dudosos: {int(baja_confianza.sum())} por confianza, {int(discrepantes.sum())} "
          f"por clustering -> {len(dudosos)} a arbitraje (tope {args.max_arbitraje})")

    # Paso 11: arbitraje con el LLM.
    dev, cambios = arbitrar_con_llm(dev, dudosos.index.tolist(),
                                    args.modelo_etiquetador, args.proveedor, args.batch)

    # Paso 12: consolidar el 80% etiquetado.
    columnas_extra = ["etiqueta_modelo", "confianza_modelo", "cluster"]
    semilla_out = semilla.copy()
    for col in columnas_extra:
        semilla_out[col] = ""
    completo = pd.concat([semilla_out, dev], ignore_index=True)
    completo.to_csv(SPLITS_DIR / "dev_etiquetado_completo.csv", index=False, encoding="utf-8-sig")

    pureza.to_csv(REPORT_DIR / "pureza_clusters.csv", index=False, encoding="utf-8-sig")
    cambios.to_csv(REPORT_DIR / "cambios_arbitraje.csv", index=False, encoding="utf-8-sig")
    dist = completo["sentimiento_v2"].value_counts().rename_axis("clase").reset_index(name="cantidad")
    dist.to_csv(REPORT_DIR / "distribucion_final_dev.csv", index=False, encoding="utf-8-sig")

    resumen = pd.DataFrame([
        ("modelo_self_training", nombre),
        ("config", json.dumps(config)),
        ("filas_semilla", len(semilla)),
        ("filas_dev_resto", len(dev)),
        ("dudosos_confianza", int(baja_confianza.sum())),
        ("dudosos_clustering", int(discrepantes.sum())),
        ("enviados_arbitraje", len(dudosos)),
        ("cambiados_por_arbitraje", len(cambios)),
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
