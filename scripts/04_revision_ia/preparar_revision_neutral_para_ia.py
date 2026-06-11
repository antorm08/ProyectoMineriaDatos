import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado_final.csv"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_neutral_para_ia.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "reporte_revision_neutral_para_ia.csv"

PALABRAS_NEUTRALES = [
    "regular",
    "normal",
    "aceptable",
    "promedio",
    "cumple",
    "mas o menos",
    "más o menos",
    "pero",
    "aunque",
    "sin embargo",
    "podria mejorar",
    "podría mejorar",
]

COLUMNAS_SALIDA = [
    "id_revision_neutral",
    "comentario_limpio",
    "estrellas",
    "sentimiento_estrella",
    "sentimiento_modelo",
    "confianza_modelo",
    "prob_neg",
    "prob_neu",
    "prob_pos",
    "sentimiento_final_provisional",
    "puntaje_neutral",
    "criterios_neutral",
    "sentimiento_ia",
    "confianza_ia",
    "justificacion_ia",
    "usar_etiqueta_ia",
]


def contiene_patron_neutral(texto):
    texto = str(texto).lower()
    return any(palabra in texto for palabra in PALABRAS_NEUTRALES)


def criterios_neutral(fila):
    criterios = []
    if fila["estrellas"] == 3:
        criterios.append("estrella_3")
    if fila["sentimiento_modelo"] == "NEU":
        criterios.append("modelo_neu")
    if fila["sentimiento_final_provisional"] == "neutral":
        criterios.append("provisional_neutral")
    if fila["patron_neutral"]:
        criterios.append("patron_textual_neutral")
    return ";".join(criterios)


def puntaje_neutral(fila):
    puntaje = 0
    if fila["estrellas"] == 3:
        puntaje += 4
    if fila["sentimiento_modelo"] == "NEU":
        puntaje += 4
    if fila["sentimiento_final_provisional"] == "neutral":
        puntaje += 3
    if fila["patron_neutral"]:
        puntaje += 2
    puntaje += float(fila["prob_neu"])
    return puntaje


def preparar_revision_neutral(input_file, output_file, report_file, max_filas):
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el dataset final integrado: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file).fillna("")
    requeridas = [
        "comentario_limpio",
        "estrellas",
        "sentimiento_estrella",
        "sentimiento_modelo",
        "confianza_modelo",
        "prob_neg",
        "prob_neu",
        "prob_pos",
        "sentimiento_final",
        "sentimiento_final_provisional",
    ]
    faltantes = [columna for columna in requeridas if columna not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    df["estrellas"] = pd.to_numeric(df["estrellas"], errors="coerce")
    for columna in ["confianza_modelo", "prob_neg", "prob_neu", "prob_pos"]:
        df[columna] = pd.to_numeric(df[columna], errors="coerce").fillna(0.0)

    candidatos = df[df["sentimiento_final"].astype(str).str.strip() == ""].copy()
    candidatos["patron_neutral"] = candidatos["comentario_limpio"].map(contiene_patron_neutral)
    candidatos = candidatos[
        (candidatos["estrellas"] == 3)
        | (candidatos["sentimiento_modelo"] == "NEU")
        | (candidatos["sentimiento_final_provisional"] == "neutral")
        | candidatos["patron_neutral"]
    ].copy()

    candidatos["criterios_neutral"] = candidatos.apply(criterios_neutral, axis=1)
    candidatos["puntaje_neutral"] = candidatos.apply(puntaje_neutral, axis=1)
    candidatos = candidatos.sort_values(
        by=["puntaje_neutral", "prob_neu", "confianza_modelo"],
        ascending=[False, False, False],
    )
    candidatos.insert(0, "id_revision_neutral", range(1, len(candidatos) + 1))

    for columna in ["sentimiento_ia", "confianza_ia", "justificacion_ia", "usar_etiqueta_ia"]:
        candidatos[columna] = ""

    salida = candidatos[COLUMNAS_SALIDA]
    if max_filas > 0:
        salida = salida.head(max_filas)

    salida.to_csv(output_file, index=False, encoding="utf-8-sig")

    reporte = pd.DataFrame(
        [
            ("filas_sin_etiqueta_final", int((df["sentimiento_final"].astype(str).str.strip() == "").sum())),
            ("candidatos_neutral", len(candidatos)),
            ("filas_exportadas", len(salida)),
            ("estrella_3", int((candidatos["estrellas"] == 3).sum())),
            ("modelo_neu", int((candidatos["sentimiento_modelo"] == "NEU").sum())),
            ("patron_textual_neutral", int(candidatos["patron_neutral"].sum())),
        ],
        columns=["metrica", "valor"],
    )
    reporte.to_csv(report_file, index=False, encoding="utf-8-sig")

    print(f"Archivo salida para IA neutral: {output_file}")
    print(f"Reporte salida: {report_file}")
    print(f"Candidatos neutral encontrados: {len(candidatos)}")
    print(f"Filas exportadas: {len(salida)}")
    print("\nCriterios mas frecuentes:")
    print(candidatos["criterios_neutral"].value_counts().head(10).to_string())


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Prepara un lote enfocado en posibles neutrales para IA.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--reporte", type=Path, default=REPORT_FILE)
    parser.add_argument(
        "--max-filas",
        type=int,
        default=300,
        help="Numero maximo de filas a exportar. 0 exporta todas.",
    )
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    preparar_revision_neutral(args.input, args.output, args.reporte, args.max_filas)


if __name__ == "__main__":
    main()
