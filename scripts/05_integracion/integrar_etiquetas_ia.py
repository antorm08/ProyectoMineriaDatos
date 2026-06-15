import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _comun.dataframes import mapear_ia_a_dataset, validar_columnas  # noqa: E402
from _comun.texto import normalizar_comparacion as normalizar_texto  # noqa: E402

DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado.csv"
REVISION_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_prioritaria_etiquetas.csv"
IA_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_prioritaria_etiquetas_ia.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado_final.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "04_integracion" / "reporte_integracion_ia.csv"
DISTRIBUTION_FILE = PROJECT_ROOT / "reports" / "04_integracion" / "distribucion_sentimiento_final_integrado.csv"

CLASES_IA_ACEPTADAS = {"negativo", "neutral"}
CONFIANZAS_IA_ACEPTADAS = {"alta", "media"}


def integrar_etiquetas_ia(dataset_file, revision_file, ia_file, output_file, report_file, distribution_file):
    for path, nombre in [
        (dataset_file, "dataset etiquetado"),
        (revision_file, "revision prioritaria"),
        (ia_file, "etiquetas IA"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"No existe {nombre}: {path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    distribution_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(dataset_file).fillna("")
    revision = pd.read_csv(revision_file).fillna("")
    ia = pd.read_csv(ia_file).fillna("")

    columnas_mapeo = ["comentario_limpio", "empresa", "sede", "estrellas"]
    validar_columnas(dataset, ["sentimiento_final", *columnas_mapeo], "dataset etiquetado")
    validar_columnas(revision, ["id_revision", *columnas_mapeo], "revision prioritaria")
    validar_columnas(
        ia,
        ["id_revision", "sentimiento_ia", "confianza_ia", "justificacion_ia", "usar_etiqueta_ia"],
        "etiquetas IA",
    )

    ia = mapear_ia_a_dataset(
        dataset,
        revision,
        ia,
        columnas_mapeo=columnas_mapeo,
        columna_id="id_revision",
        nombre_base="revision prioritaria",
        nombre_ia="etiquetas IA",
    )

    dataset["sentimiento_final_original"] = dataset["sentimiento_final"]
    dataset["sentimiento_final_origen"] = dataset["sentimiento_final"].apply(
        lambda valor: "consistencia_estrella_modelo" if str(valor).strip() else "sin_etiqueta"
    )
    for columna in ["sentimiento_ia", "confianza_ia", "justificacion_ia", "usar_etiqueta_ia"]:
        if columna not in dataset.columns:
            dataset[columna] = ""

    integradas = 0
    rechazadas = 0
    for _, fila in ia.iterrows():
        dataset_index = int(fila["dataset_index"])
        sentimiento_ia = normalizar_texto(fila["sentimiento_ia"])
        confianza_ia = normalizar_texto(fila["confianza_ia"])
        usar_etiqueta_ia = normalizar_texto(fila["usar_etiqueta_ia"])

        dataset.at[dataset_index, "sentimiento_ia"] = sentimiento_ia
        dataset.at[dataset_index, "confianza_ia"] = confianza_ia
        dataset.at[dataset_index, "justificacion_ia"] = fila["justificacion_ia"]
        dataset.at[dataset_index, "usar_etiqueta_ia"] = usar_etiqueta_ia

        cumple_regla = (
            str(dataset.at[dataset_index, "sentimiento_final"]).strip() == ""
            and usar_etiqueta_ia == "si"
            and confianza_ia in CONFIANZAS_IA_ACEPTADAS
            and sentimiento_ia in CLASES_IA_ACEPTADAS
        )

        if cumple_regla:
            dataset.at[dataset_index, "sentimiento_final"] = sentimiento_ia
            dataset.at[dataset_index, "sentimiento_final_origen"] = "ia_clase_minoritaria"
            integradas += 1
        else:
            rechazadas += 1

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
            ("filas_ia", len(ia)),
            ("etiquetas_ia_integradas", integradas),
            ("etiquetas_ia_no_integradas", rechazadas),
            ("finales_con_etiqueta", int((dataset["sentimiento_final"].astype(str).str.strip() != "").sum())),
            ("finales_sin_etiqueta", int((dataset["sentimiento_final"].astype(str).str.strip() == "").sum())),
        ],
        columns=["metrica", "valor"],
    )
    reporte.to_csv(report_file, index=False, encoding="utf-8-sig")

    print(f"Archivo final generado: {output_file}")
    print(f"Reporte generado: {report_file}")
    print(f"Distribucion generada: {distribution_file}")
    print(f"Etiquetas IA integradas: {integradas}")
    print(f"Etiquetas IA no integradas: {rechazadas}")
    print("\nDistribucion final integrada:")
    print(distribucion.to_string(index=False))


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Integra etiquetas IA conservadoras al dataset final.")
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--revision", type=Path, default=REVISION_FILE)
    parser.add_argument("--ia", type=Path, default=IA_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--reporte", type=Path, default=REPORT_FILE)
    parser.add_argument("--distribucion", type=Path, default=DISTRIBUTION_FILE)
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    integrar_etiquetas_ia(
        args.dataset,
        args.revision,
        args.ia,
        args.output,
        args.reporte,
        args.distribucion,
    )


if __name__ == "__main__":
    main()
