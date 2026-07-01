"""Prepara el flujo de etiquetado manual inicial.

Separa el dataset en 80% desarrollo y 20% prueba final. Del 80% toma una
muestra de 500 registros para etiquetado manual y deja el resto listo para ser
etiquetado automaticamente por el modelo inicial.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_limpio.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "manual_500"

COLUMNAS_TEXTO = {"comentario_limpio", "texto_modelo"}


def preparar_muestra(input_file, output_dir, sample_size, test_size, random_state):
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el dataset de entrada: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_file).fillna("")
    faltantes = COLUMNAS_TEXTO - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    df = df[df["texto_modelo"].astype(str).str.strip() != ""].copy()
    if len(df) <= sample_size:
        raise ValueError(f"El dataset tiene {len(df)} filas; debe superar sample_size={sample_size}.")

    estratificar = None
    if "sentimiento_estrella" in df.columns and df["sentimiento_estrella"].nunique() > 1:
        estratificar = df["sentimiento_estrella"]

    desarrollo_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=estratificar,
    )

    estratificar_desarrollo = None
    if "sentimiento_estrella" in desarrollo_df.columns and desarrollo_df["sentimiento_estrella"].nunique() > 1:
        estratificar_desarrollo = desarrollo_df["sentimiento_estrella"]

    muestra_df, resto_df = train_test_split(
        desarrollo_df,
        train_size=sample_size,
        random_state=random_state,
        stratify=estratificar_desarrollo,
    )

    muestra_df = muestra_df.copy()
    muestra_df.insert(0, "id_manual", range(1, len(muestra_df) + 1))
    muestra_df["etiqueta_manual"] = ""
    muestra_df["observacion_manual"] = ""

    desarrollo_df.to_csv(output_dir / "desarrollo_80.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(output_dir / "test_20_reservado.csv", index=False, encoding="utf-8-sig")
    muestra_df.to_csv(output_dir / "muestra_500_para_etiquetar.csv", index=False, encoding="utf-8-sig")
    resto_df.to_csv(output_dir / "resto_80_para_etiquetar.csv", index=False, encoding="utf-8-sig")

    reporte = pd.DataFrame(
        [
            ("filas_dataset_base", len(df)),
            ("filas_desarrollo_80", len(desarrollo_df)),
            ("filas_test_20_reservado", len(test_df)),
            ("filas_muestra_manual", len(muestra_df)),
            ("filas_resto_para_etiquetar", len(resto_df)),
            ("test_size", test_size),
            ("sample_size", sample_size),
            ("random_state", random_state),
            ("estratificacion", "sentimiento_estrella" if estratificar is not None else "sin_estratificar"),
        ],
        columns=["metrica", "valor"],
    )
    reporte.to_csv(output_dir / "reporte_preparacion_manual_500.csv", index=False, encoding="utf-8-sig")

    print(f"Desarrollo 80%: {output_dir / 'desarrollo_80.csv'}")
    print(f"Test 20% reservado: {output_dir / 'test_20_reservado.csv'}")
    print(f"Muestra para etiquetar: {output_dir / 'muestra_500_para_etiquetar.csv'}")
    print(f"Resto para etiquetar: {output_dir / 'resto_80_para_etiquetar.csv'}")
    print("Llena la columna 'etiqueta_manual' antes de continuar.")


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Separa 80/20 y extrae 500 registros para etiquetado manual.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    preparar_muestra(args.input, args.output_dir, args.sample_size, args.test_size, args.random_state)


if __name__ == "__main__":
    main()
