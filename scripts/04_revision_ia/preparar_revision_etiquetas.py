import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado.csv"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_prioritaria_etiquetas.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "reporte_revision_prioritaria.csv"

COLUMNAS_SALIDA = [
    "id_revision",
    "prioridad_revision",
    "tipo_revision",
    "comentario",
    "comentario_limpio",
    "empresa",
    "sede",
    "rubro",
    "estrellas",
    "sentimiento_estrella",
    "sentimiento_modelo",
    "confianza_modelo",
    "prob_neg",
    "prob_neu",
    "prob_pos",
    "sentimiento_final",
    "sentimiento_final_provisional",
    "confianza_etiqueta",
    "motivo_revision_etiqueta",
    "sentimiento_manual",
    "observacion_manual",
]


def asignar_tipo_revision(fila):
    estrellas = fila["estrellas"]
    modelo = fila["sentimiento_modelo"]
    confianza = fila["confianza_modelo"]
    final = fila["sentimiento_final"]

    if final != "":
        return "revision_por_confianza_o_limpieza"
    if estrellas == 3:
        return "posible_neutral_ambiguo"
    if estrellas == 2:
        return "posible_negativo_ambiguo"
    if confianza >= 0.80 and final == "":
        return "contradiccion_alta_confianza"
    if modelo == "NEU":
        return "modelo_detecta_neutral"
    if modelo == "NEG":
        return "modelo_detecta_negativo"

    return "otra_inconsistencia"


def asignar_prioridad(fila):
    tipo = fila["tipo_revision"]
    estrellas = fila["estrellas"]
    confianza = fila["confianza_modelo"]
    modelo = fila["sentimiento_modelo"]

    if estrellas == 3 or tipo == "posible_neutral_ambiguo":
        return 1
    if estrellas == 2 or tipo == "posible_negativo_ambiguo":
        return 2
    if confianza >= 0.80 and tipo == "contradiccion_alta_confianza":
        return 3
    if modelo in ["NEU", "NEG"]:
        return 4

    return 5


def preparar_revision(input_file, output_file, report_file, max_filas):
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el archivo etiquetado: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file).fillna("")
    if "requiere_revision_etiqueta" not in df.columns:
        raise ValueError("Falta la columna requiere_revision_etiqueta.")

    df["estrellas"] = pd.to_numeric(df["estrellas"], errors="coerce")
    df["confianza_modelo"] = pd.to_numeric(df["confianza_modelo"], errors="coerce").fillna(0.0)

    revision = df[df["requiere_revision_etiqueta"].astype(bool)].copy()
    revision["tipo_revision"] = revision.apply(asignar_tipo_revision, axis=1)
    revision["prioridad_revision"] = revision.apply(asignar_prioridad, axis=1)
    revision = revision.sort_values(
        by=["prioridad_revision", "confianza_modelo"],
        ascending=[True, False],
    )
    revision.insert(0, "id_revision", range(1, len(revision) + 1))

    revision["sentimiento_manual"] = ""
    revision["observacion_manual"] = ""
    columnas_existentes = [columna for columna in COLUMNAS_SALIDA if columna in revision.columns]

    if max_filas > 0:
        revision_salida = revision[columnas_existentes].head(max_filas)
    else:
        revision_salida = revision[columnas_existentes]

    revision_salida.to_csv(output_file, index=False, encoding="utf-8-sig")

    reporte = pd.DataFrame(
        [
            ("filas_revision_total", len(revision)),
            ("filas_exportadas", len(revision_salida)),
            ("prioridad_1", int((revision["prioridad_revision"] == 1).sum())),
            ("prioridad_2", int((revision["prioridad_revision"] == 2).sum())),
            ("prioridad_3", int((revision["prioridad_revision"] == 3).sum())),
            ("prioridad_4", int((revision["prioridad_revision"] == 4).sum())),
            ("prioridad_5", int((revision["prioridad_revision"] == 5).sum())),
        ],
        columns=["metrica", "valor"],
    )
    reporte.to_csv(report_file, index=False, encoding="utf-8-sig")

    print(f"Archivo generado: {output_file}")
    print(f"Reporte generado: {report_file}")
    print(f"Filas totales para revision: {len(revision)}")
    print(f"Filas exportadas: {len(revision_salida)}")
    print("\nDistribucion por prioridad:")
    print(revision["prioridad_revision"].value_counts().sort_index().to_string())
    print("\nDistribucion por tipo de revision:")
    print(revision["tipo_revision"].value_counts().to_string())
    print("\nDistribucion por estrellas en revision:")
    print(revision["estrellas"].value_counts().sort_index().to_string())


def obtener_argumentos():
    parser = argparse.ArgumentParser(
        description="Prepara un CSV priorizado para revision manual de etiquetas."
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--reporte", type=Path, default=REPORT_FILE)
    parser.add_argument(
        "--max-filas",
        type=int,
        default=500,
        help="Numero maximo de filas a exportar. Usa 0 para exportar todas.",
    )
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    preparar_revision(args.input, args.output, args.reporte, args.max_filas)


if __name__ == "__main__":
    main()
