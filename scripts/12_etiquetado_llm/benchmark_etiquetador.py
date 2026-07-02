"""Fase 12 (previa): benchmark de modelos LLM candidatos para el etiquetado asistido.

Antes de etiquetar la semilla de 500 y el conjunto de prueba, se elige el modelo
anotador comparando candidatos sobre una muestra estratificada de resenas que YA
tienen etiqueta consolidada de confianza alta (consenso estrellas+modelo del pipeline
de supervision debil). Metricas por candidato:

    - acuerdo exacto     (etiqueta LLM == etiqueta consolidada)
    - acuerdo adyacente  (|posicion ordinal LLM - consolidada| <= 1)
    - cobertura          (proporcion de respuestas validas)
    - latencia total

El acuerdo exacto con etiquetas de consenso es un proxy razonable de calidad de
anotacion; el acuerdo adyacente tolera la ambiguedad natural entre clases vecinas
(positivo vs muy positivo). El ganador se usa en etiquetar_llm.py.

Salida: reports/11_etiquetado_llm/benchmark_etiquetador.csv
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
from _comun.llm import CLASES, etiquetar_lote  # noqa: E402

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado_final.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "11_etiquetado_llm"

# Candidatos: (nombre corto, proveedor, id de modelo)
CANDIDATOS = [
    ("qwen3.5-122b", "nvidia", "qwen/qwen3.5-122b-a10b"),
    ("qwen3.5-397b", "nvidia", "qwen/qwen3.5-397b-a17b"),
    ("mistral-large-3", "nvidia", "mistralai/mistral-large-3-675b-instruct-2512"),
    ("gpt-oss-120b", "nvidia", "openai/gpt-oss-120b"),
    ("llama-3.3-70b", "nvidia", "meta/llama-3.3-70b-instruct"),
    ("gpt-4o-mini", "openrouter", "openai/gpt-4o-mini"),
]

IDX = {c: i for i, c in enumerate(CLASES)}


def muestra_referencia(por_clase, random_state):
    """Muestra estratificada de resenas con etiqueta consolidada de confianza alta."""
    df = pd.read_csv(INPUT_FILE).fillna("")
    df = df[(df["texto_modelo"].astype(str).str.strip() != "")
            & (df["sentimiento_final"].isin(CLASES))
            & (df["confianza_etiqueta"] == "alta")]
    partes = [
        df[df["sentimiento_final"] == clase].sample(
            n=min(por_clase, (df["sentimiento_final"] == clase).sum()),
            random_state=random_state,
        )
        for clase in CLASES
    ]
    muestra = pd.concat(partes).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return muestra[["comentario_limpio", "sentimiento_final"]]


def evaluar_candidato(nombre, proveedor, modelo, muestra, batch):
    textos = muestra["comentario_limpio"].tolist()
    reales = muestra["sentimiento_final"].tolist()
    inicio = time.time()
    resultados = []
    try:
        for i in range(0, len(textos), batch):
            lote = textos[i:i + batch]
            resultados.extend(etiquetar_lote(lote, modelo=modelo, proveedor=proveedor,
                                             ids=list(range(i, i + len(lote)))))
    except Exception as exc:  # noqa: BLE001 - un candidato caido no debe frenar el benchmark
        print(f"   FALLO: {exc}")
        return {"candidato": nombre, "proveedor": proveedor, "modelo": modelo,
                "acuerdo_exacto": None, "acuerdo_adyacente": None,
                "cobertura": 0.0, "latencia_s": round(time.time() - inicio, 1),
                "error": str(exc)[:150]}
    latencia = time.time() - inicio

    validos = exactos = adyacentes = 0
    for real, res in zip(reales, resultados):
        if res is None:
            continue
        validos += 1
        exactos += int(res["etiqueta"] == real)
        adyacentes += int(abs(IDX[res["etiqueta"]] - IDX[real]) <= 1)
    n = len(reales)
    return {
        "candidato": nombre, "proveedor": proveedor, "modelo": modelo,
        "acuerdo_exacto": round(exactos / validos, 4) if validos else None,
        "acuerdo_adyacente": round(adyacentes / validos, 4) if validos else None,
        "cobertura": round(validos / n, 4),
        "latencia_s": round(latencia, 1),
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark de LLMs anotadores (fase 12).")
    parser.add_argument("--por-clase", type=int, default=8)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    muestra = muestra_referencia(args.por_clase, args.random_state)
    print(f"Muestra de referencia: {len(muestra)} resenas "
          f"({args.por_clase} por clase, confianza alta, consenso)")

    filas = []
    for nombre, proveedor, modelo in CANDIDATOS:
        print(f"\n>> {nombre} ({proveedor}: {modelo})")
        fila = evaluar_candidato(nombre, proveedor, modelo, muestra, args.batch)
        filas.append(fila)
        if fila["acuerdo_exacto"] is not None:
            print(f"   exacto {fila['acuerdo_exacto']:.2%} | adyacente {fila['acuerdo_adyacente']:.2%} "
                  f"| cobertura {fila['cobertura']:.0%} | {fila['latencia_s']}s")

    tabla = pd.DataFrame(filas).sort_values("acuerdo_exacto", ascending=False, na_position="last")
    tabla.to_csv(REPORT_DIR / "benchmark_etiquetador.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 78)
    print(tabla.drop(columns=["error"]).to_string(index=False))
    print(f"\nGuardado en: {REPORT_DIR / 'benchmark_etiquetador.csv'}")


if __name__ == "__main__":
    main()
