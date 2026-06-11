import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_prioritaria_etiquetas.csv"
OUTPUT_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_prioritaria_para_ia.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "reporte_revision_para_ia.csv"

COLUMNAS_ENTRADA = [
    "id_revision",
    "comentario_limpio",
    "estrellas",
    "sentimiento_estrella",
    "sentimiento_modelo",
    "confianza_modelo",
    "prob_neg",
    "prob_neu",
    "prob_pos",
    "sentimiento_final_provisional",
]

COLUMNAS_IA = [
    "sentimiento_ia",
    "confianza_ia",
    "justificacion_ia",
    "usar_etiqueta_ia",
]


def preparar_revision_para_ia(input_file, output_file, report_file, max_filas):
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el archivo de revision: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file).fillna("")
    faltantes = [columna for columna in COLUMNAS_ENTRADA if columna not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    salida = df[COLUMNAS_ENTRADA].copy()
    if max_filas > 0:
        salida = salida.head(max_filas)

    for columna in COLUMNAS_IA:
        salida[columna] = ""

    salida.to_csv(output_file, index=False, encoding="utf-8-sig")

    reporte = pd.DataFrame(
        [
            ("filas_entrada", len(df)),
            ("filas_exportadas", len(salida)),
            ("columnas_exportadas", len(salida.columns)),
        ],
        columns=["metrica", "valor"],
    )
    reporte.to_csv(report_file, index=False, encoding="utf-8-sig")

    print(f"Archivo entrada: {input_file}")
    print(f"Archivo salida para IA: {output_file}")
    print(f"Reporte salida: {report_file}")
    print(f"Filas entrada: {len(df)}")
    print(f"Filas exportadas: {len(salida)}")
    print("\nColumnas exportadas:")
    for columna in salida.columns:
        print(f"- {columna}")


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Prepara un CSV reducido para etiquetado asistido por IA.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--reporte", type=Path, default=REPORT_FILE)
    parser.add_argument(
        "--max-filas",
        type=int,
        default=0,
        help="Numero maximo de filas a exportar. 0 exporta todas.",
    )
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    preparar_revision_para_ia(args.input, args.output, args.reporte, args.max_filas)


if __name__ == "__main__":
    main()
