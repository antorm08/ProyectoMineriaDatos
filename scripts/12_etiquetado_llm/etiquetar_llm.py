"""Fase 12: etiquetado asistido por LLM de la semilla (500) y del test (20%).

El docente pidio etiquetar manualmente ~500 registros representativos. El equipo
automatiza ese paso con un LLM externo como anotador asistido (human-in-the-loop):

    1. El LLM (elegido en benchmark_etiquetador.py) etiqueta cada resena con una de
       las 5 clases ordinales, una confianza y una justificacion breve.
    2. Se genera un CSV de revision para el equipo ORDENADO por confianza ascendente,
       con columna `etiqueta_corregida` para correcciones manuales rapidas.
    3. Se mide el acuerdo del LLM con las etiquetas del pipeline anterior de
       supervision debil (feedback del docente: "coincidimos si esta igual que el
       manual") y con la senal de estrellas.

El conjunto de prueba (960) se etiqueta con el MISMO protocolo pero de forma
independiente: sus etiquetas son la referencia (ground truth) de la evaluacion final
de la fase 15 y nunca alimentan el entrenamiento.

El progreso se cachea en data/splits_v2/cache_etiquetas_llm.csv: si la corrida se
interrumpe (rate limit, red), al relanzar continua donde quedo.

Entradas:
    data/splits_v2/semilla.csv, data/splits_v2/test.csv

Salidas:
    data/splits_v2/semilla_etiquetada.csv     (con sentimiento_v2)
    data/splits_v2/test_etiquetado.csv        (con sentimiento_v2)
    reports/11_etiquetado_llm/revision_equipo_semilla.csv
    reports/11_etiquetado_llm/acuerdo_vs_pipeline_anterior.csv
    reports/11_etiquetado_llm/distribucion_etiquetas_llm.csv
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
from _comun.llm import CLASES, etiquetar_lote  # noqa: E402

SPLITS_DIR = PROJECT_ROOT / "data" / "splits_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "11_etiquetado_llm"
CACHE_FILE = SPLITS_DIR / "cache_etiquetas_llm.csv"

IDX = {c: i for i, c in enumerate(CLASES)}


def cargar_cache():
    if CACHE_FILE.exists():
        cache = pd.read_csv(CACHE_FILE)
        return {(fila.bloque, fila.id_registro): fila for fila in cache.itertuples()}
    return {}


def anexar_cache(filas):
    df = pd.DataFrame(filas)
    df.to_csv(CACHE_FILE, mode="a", header=not CACHE_FILE.exists(),
              index=False, encoding="utf-8-sig")


def etiquetar_bloque(nombre, df, modelo, proveedor, batch, cache, pausa):
    """Etiqueta las filas de un bloque que no esten en cache. Devuelve el df anotado."""
    pendientes = df[~df["id_registro"].map(lambda i: (nombre, i) in cache)]
    print(f"\n== Bloque {nombre}: {len(df)} filas ({len(pendientes)} pendientes, "
          f"{len(df) - len(pendientes)} en cache)")

    hechos = 0
    for i in range(0, len(pendientes), batch):
        lote = pendientes.iloc[i:i + batch]
        ids = lote["id_registro"].tolist()
        resultados = etiquetar_lote(lote["comentario_limpio"].tolist(), modelo=modelo,
                                    proveedor=proveedor, ids=ids)
        # Reintento individual de los que el LLM omitio o devolvio invalido.
        for j, (id_reg, res) in enumerate(zip(ids, resultados)):
            if res is None:
                reintento = etiquetar_lote([lote["comentario_limpio"].iloc[j]],
                                           modelo=modelo, proveedor=proveedor, ids=[id_reg])
                resultados[j] = reintento[0]

        filas_cache = []
        for id_reg, res in zip(ids, resultados):
            filas_cache.append({
                "bloque": nombre,
                "id_registro": id_reg,
                "etiqueta_llm": res["etiqueta"] if res else "",
                "confianza_llm": res["confianza"] if res else 0.0,
                "justificacion_llm": res["justificacion"] if res else "sin respuesta valida",
                "modelo": modelo,
            })
        anexar_cache(filas_cache)
        for fila in filas_cache:
            cache[(nombre, fila["id_registro"])] = pd.Series(fila)

        hechos += len(lote)
        print(f"   {min(hechos, len(pendientes))}/{len(pendientes)} etiquetadas", flush=True)
        time.sleep(pausa)

    anot = pd.DataFrame([{
        "id_registro": id_reg,
        "etiqueta_llm": cache[(nombre, id_reg)].etiqueta_llm
        if hasattr(cache[(nombre, id_reg)], "etiqueta_llm") else cache[(nombre, id_reg)]["etiqueta_llm"],
        "confianza_llm": cache[(nombre, id_reg)].confianza_llm
        if hasattr(cache[(nombre, id_reg)], "confianza_llm") else cache[(nombre, id_reg)]["confianza_llm"],
        "justificacion_llm": cache[(nombre, id_reg)].justificacion_llm
        if hasattr(cache[(nombre, id_reg)], "justificacion_llm") else cache[(nombre, id_reg)]["justificacion_llm"],
    } for id_reg in df["id_registro"]])
    salida = df.merge(anot, on="id_registro", how="left")
    salida["sentimiento_v2"] = salida["etiqueta_llm"]
    return salida


def acuerdo(df, columna_referencia):
    """Acuerdo exacto y adyacente de etiqueta_llm frente a una columna de referencia."""
    con_ref = df[df[columna_referencia].isin(CLASES) & df["etiqueta_llm"].isin(CLASES)]
    if len(con_ref) == 0:
        return None, None, 0
    exacto = (con_ref["etiqueta_llm"] == con_ref[columna_referencia]).mean()
    ady = (abs(con_ref["etiqueta_llm"].map(IDX) - con_ref[columna_referencia].map(IDX)) <= 1).mean()
    return round(float(exacto), 4), round(float(ady), 4), len(con_ref)


def main():
    parser = argparse.ArgumentParser(description="Etiquetado LLM de semilla y test (fase 12).")
    parser.add_argument("--modelo", required=True, help="Id del modelo (p. ej. qwen/qwen3.5-122b-a10b)")
    parser.add_argument("--proveedor", default="nvidia", choices=["nvidia", "openrouter"])
    parser.add_argument("--batch", type=int, default=15)
    parser.add_argument("--pausa", type=float, default=0.5, help="Segundos entre lotes.")
    parser.add_argument("--bloques", nargs="+", default=["semilla", "test"],
                        choices=["semilla", "test"])
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cache = cargar_cache()

    bloques = {}
    for nombre in args.bloques:
        df = pd.read_csv(SPLITS_DIR / f"{nombre}.csv").fillna("")
        bloques[nombre] = etiquetar_bloque(nombre, df, args.modelo, args.proveedor,
                                           args.batch, cache, args.pausa)
        bloques[nombre].to_csv(SPLITS_DIR / f"{nombre}_etiquetada.csv"
                               if nombre == "semilla" else SPLITS_DIR / f"{nombre}_etiquetado.csv",
                               index=False, encoding="utf-8-sig")

    # CSV de revision para el equipo (solo semilla), ordenado por confianza ascendente.
    if "semilla" in bloques:
        semilla = bloques["semilla"]
        revision = semilla[[
            "id_registro", "empresa", "rubro", "estrellas", "comentario_limpio",
            "etiqueta_llm", "confianza_llm", "justificacion_llm", "sentimiento_final",
        ]].rename(columns={"sentimiento_final": "etiqueta_pipeline_anterior"})
        revision = revision.sort_values("confianza_llm")
        revision["etiqueta_corregida"] = ""
        revision.to_csv(REPORT_DIR / "revision_equipo_semilla.csv",
                        index=False, encoding="utf-8-sig")

    # Acuerdo con el pipeline anterior y con las estrellas (donde existan).
    filas_acuerdo = []
    for nombre, df in bloques.items():
        for referencia in ["sentimiento_final", "sentimiento_estrella"]:
            exacto, ady, n = acuerdo(df, referencia)
            filas_acuerdo.append({
                "bloque": nombre, "referencia": referencia, "n_comparadas": n,
                "acuerdo_exacto": exacto, "acuerdo_adyacente": ady,
            })
    pd.DataFrame(filas_acuerdo).to_csv(REPORT_DIR / "acuerdo_vs_pipeline_anterior.csv",
                                       index=False, encoding="utf-8-sig")

    # Distribucion de clases resultante.
    dist = pd.concat([
        df["etiqueta_llm"].value_counts().rename(nombre) for nombre, df in bloques.items()
    ], axis=1).fillna(0).astype(int)
    dist.to_csv(REPORT_DIR / "distribucion_etiquetas_llm.csv", encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("ACUERDO CON REFERENCIAS")
    print(pd.DataFrame(filas_acuerdo).to_string(index=False))
    print("\nDISTRIBUCION DE ETIQUETAS LLM")
    print(dist.to_string())
    print(f"\nRevision del equipo: {REPORT_DIR / 'revision_equipo_semilla.csv'}")


if __name__ == "__main__":
    main()
