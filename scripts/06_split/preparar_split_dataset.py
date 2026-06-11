import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado_final.csv"
TRAIN_FILE = PROJECT_ROOT / "data" / "splits" / "train.csv"
VALID_FILE = PROJECT_ROOT / "data" / "splits" / "valid.csv"
TEST_FILE = PROJECT_ROOT / "data" / "splits" / "test.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "05_split" / "reporte_split_dataset.csv"
TRAIN_DISTRIBUTION_FILE = PROJECT_ROOT / "reports" / "05_split" / "distribucion_split_train.csv"
VALID_DISTRIBUTION_FILE = PROJECT_ROOT / "reports" / "05_split" / "distribucion_split_valid.csv"
TEST_DISTRIBUTION_FILE = PROJECT_ROOT / "reports" / "05_split" / "distribucion_split_test.csv"

COLUMNAS_REQUERIDAS = {"comentario_limpio", "texto_modelo", "sentimiento_final"}


def guardar_distribucion(df, output_file):
    distribucion = (
        df["sentimiento_final"]
        .value_counts()
        .rename_axis("sentimiento_final")
        .reset_index(name="cantidad")
    )
    distribucion["porcentaje"] = (distribucion["cantidad"] / len(df) * 100).round(2)
    distribucion.to_csv(output_file, index=False, encoding="utf-8-sig")
    return distribucion


def preparar_split(input_file, train_file, valid_file, test_file, report_file, random_state):
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el dataset final: {input_file}")

    for path in [train_file, valid_file, test_file, report_file]:
        path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file).fillna("")
    faltantes = COLUMNAS_REQUERIDAS - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    df_modelo = df[df["sentimiento_final"].astype(str).str.strip() != ""].copy()
    df_modelo = df_modelo[df_modelo["texto_modelo"].astype(str).str.strip() != ""].copy()

    train_df, temp_df = train_test_split(
        df_modelo,
        test_size=0.30,
        random_state=random_state,
        stratify=df_modelo["sentimiento_final"],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_df["sentimiento_final"],
    )

    train_df.to_csv(train_file, index=False, encoding="utf-8-sig")
    valid_df.to_csv(valid_file, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_file, index=False, encoding="utf-8-sig")

    train_dist = guardar_distribucion(train_df, TRAIN_DISTRIBUTION_FILE)
    valid_dist = guardar_distribucion(valid_df, VALID_DISTRIBUTION_FILE)
    test_dist = guardar_distribucion(test_df, TEST_DISTRIBUTION_FILE)

    reporte = pd.DataFrame(
        [
            ("filas_dataset_original", len(df)),
            ("filas_con_sentimiento_final", int((df["sentimiento_final"].astype(str).str.strip() != "").sum())),
            ("filas_usadas_modelado", len(df_modelo)),
            ("filas_train", len(train_df)),
            ("filas_valid", len(valid_df)),
            ("filas_test", len(test_df)),
            ("proporcion_train", round(len(train_df) / len(df_modelo), 4)),
            ("proporcion_valid", round(len(valid_df) / len(df_modelo), 4)),
            ("proporcion_test", round(len(test_df) / len(df_modelo), 4)),
            ("random_state", random_state),
        ],
        columns=["metrica", "valor"],
    )
    reporte.to_csv(report_file, index=False, encoding="utf-8-sig")

    print(f"Train generado: {train_file}")
    print(f"Valid generado: {valid_file}")
    print(f"Test generado: {test_file}")
    print(f"Reporte generado: {report_file}")
    print("\nFilas:")
    print(f"train: {len(train_df)}")
    print(f"valid: {len(valid_df)}")
    print(f"test: {len(test_df)}")
    print("\nDistribucion train:")
    print(train_dist.to_string(index=False))
    print("\nDistribucion valid:")
    print(valid_dist.to_string(index=False))
    print("\nDistribucion test:")
    print(test_dist.to_string(index=False))


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Prepara split estratificado train/valid/test.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--train", type=Path, default=TRAIN_FILE)
    parser.add_argument("--valid", type=Path, default=VALID_FILE)
    parser.add_argument("--test", type=Path, default=TEST_FILE)
    parser.add_argument("--reporte", type=Path, default=REPORT_FILE)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    preparar_split(args.input, args.train, args.valid, args.test, args.reporte, args.random_state)


if __name__ == "__main__":
    main()
