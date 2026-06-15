import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _comun.dataframes import mapear_ia_a_dataset, validar_columnas  # noqa: E402
from _comun.texto import normalizar_comparacion as normalizar_texto  # noqa: E402

DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado_final.csv"
NEUTRAL_BASE_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_neutral_para_ia.csv"
NEUTRAL_IA_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_neutral_etiquetas_ia.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado_final.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "04_integracion" / "reporte_integracion_ia_neutral.csv"
DISTRIBUTION_FILE = PROJECT_ROOT / "reports" / "04_integracion" / "distribucion_sentimiento_final_integrado.csv"


def integrar_neutrales(dataset_file, neutral_base_file, neutral_ia_file, output_file, report_file, distribution_file):
    for path, nombre in [
        (dataset_file, "dataset final"),
        (neutral_base_file, "lote neutral base"),
        (neutral_ia_file, "etiquetas IA neutral"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"No existe {nombre}: {path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    distribution_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(dataset_file).fillna("")
    neutral_base = pd.read_csv(neutral_base_file).fillna("")
    neutral_ia = pd.read_csv(neutral_ia_file).fillna("")

    columnas_mapeo = ["comentario_limpio", "estrellas"]
    validar_columnas(dataset, ["sentimiento_final", *columnas_mapeo], "dataset final")
    validar_columnas(neutral_base, ["id_revision_neutral", *columnas_mapeo], "lote neutral base")
    validar_columnas(
        neutral_ia,
        ["id_revision_neutral", "sentimiento_ia", "confianza_ia", "justificacion_ia", "usar_etiqueta_ia"],
        "etiquetas IA neutral",
    )

    neutral_ia = mapear_ia_a_dataset(
        dataset,
        neutral_base,
        neutral_ia,
        columnas_mapeo=columnas_mapeo,
        columna_id="id_revision_neutral",
        nombre_base="lote neutral base",
        nombre_ia="etiquetas IA neutral",
    )

    for columna in ["sentimiento_ia_neutral", "confianza_ia_neutral", "justificacion_ia_neutral", "usar_etiqueta_ia_neutral"]:
        if columna not in dataset.columns:
            dataset[columna] = ""

    integradas = 0
    no_integradas = 0
    for _, fila in neutral_ia.iterrows():
        dataset_index = int(fila["dataset_index"])
        sentimiento_ia = normalizar_texto(fila["sentimiento_ia"])
        confianza_ia = normalizar_texto(fila["confianza_ia"])
        usar_etiqueta_ia = normalizar_texto(fila["usar_etiqueta_ia"])

        dataset.at[dataset_index, "sentimiento_ia_neutral"] = sentimiento_ia
        dataset.at[dataset_index, "confianza_ia_neutral"] = confianza_ia
        dataset.at[dataset_index, "justificacion_ia_neutral"] = fila["justificacion_ia"]
        dataset.at[dataset_index, "usar_etiqueta_ia_neutral"] = usar_etiqueta_ia

        cumple_regla = (
            str(dataset.at[dataset_index, "sentimiento_final"]).strip() == ""
            and sentimiento_ia == "neutral"
            and confianza_ia in {"alta", "media"}
            and usar_etiqueta_ia == "si"
        )

        if cumple_regla:
            dataset.at[dataset_index, "sentimiento_final"] = "neutral"
            dataset.at[dataset_index, "sentimiento_final_origen"] = "ia_lote_neutral"
            integradas += 1
        else:
            no_integradas += 1

    dataset.to_csv(output_file, index=False, encoding="utf-8-sig")

    distribucion = (
        dataset.loc[dataset["sentimiento_final"].astype(str).str.strip() != "", "sentimiento_final"]
        .value_counts()
        .rename_axis("sentimiento_final")
        .reset_index(name="cantidad")
    )
    distribucion.to_csv(distribution_file, index=False, encoding="utf-8-sig")

    reporte = pd.DataFrame(
        [
            ("filas_dataset", len(dataset)),
            ("filas_ia_neutral", len(neutral_ia)),
            ("neutrales_integrados", integradas),
            ("filas_no_integradas", no_integradas),
            ("finales_con_etiqueta", int((dataset["sentimiento_final"].astype(str).str.strip() != "").sum())),
            ("finales_sin_etiqueta", int((dataset["sentimiento_final"].astype(str).str.strip() == "").sum())),
        ],
        columns=["metrica", "valor"],
    )
    reporte.to_csv(report_file, index=False, encoding="utf-8-sig")

    print(f"Archivo final actualizado: {output_file}")
    print(f"Reporte generado: {report_file}")
    print(f"Distribucion actualizada: {distribution_file}")
    print(f"Neutrales integrados: {integradas}")
    print(f"Filas no integradas: {no_integradas}")
    print("\nDistribucion final actualizada:")
    print(distribucion.to_string(index=False))


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Integra etiquetas IA del lote enfocado en neutral.")
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--neutral-base", type=Path, default=NEUTRAL_BASE_FILE)
    parser.add_argument("--neutral-ia", type=Path, default=NEUTRAL_IA_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--reporte", type=Path, default=REPORT_FILE)
    parser.add_argument("--distribucion", type=Path, default=DISTRIBUTION_FILE)
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    integrar_neutrales(
        args.dataset,
        args.neutral_base,
        args.neutral_ia,
        args.output,
        args.reporte,
        args.distribucion,
    )


if __name__ == "__main__":
    main()
