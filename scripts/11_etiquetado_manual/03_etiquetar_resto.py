"""Etiqueta automaticamente el resto del 80% usando el modelo inicial."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_FILE = PROJECT_ROOT / "models" / "modelo_etiquetador_inicial.joblib"
RESTO_FILE = PROJECT_ROOT / "data" / "manual_500" / "resto_80_para_etiquetar.csv"
MANUAL_FILE = PROJECT_ROOT / "data" / "manual_500" / "muestra_500_para_etiquetar.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "manual_500" / "desarrollo_80_etiquetado.csv"


def etiquetar(model_file, resto_file, manual_file, output_file):
    if not model_file.exists():
        raise FileNotFoundError(f"No existe el modelo inicial: {model_file}")
    modelo_guardado = joblib.load(model_file)
    modelo = modelo_guardado["modelo"]

    resto_df = pd.read_csv(resto_file).fillna("")
    manual_df = pd.read_csv(manual_file).fillna("")
    manual_df = manual_df[manual_df["etiqueta_manual"].astype(str).str.strip() != ""].copy()

    pred = modelo.predict(resto_df["texto_modelo"].astype(str))
    resto_df = resto_df.copy()
    resto_df["sentimiento_final"] = pred
    resto_df["origen_etiqueta"] = f"modelo_inicial_{modelo_guardado['nombre']}"

    manual_df["sentimiento_final"] = manual_df["etiqueta_manual"]
    manual_df["origen_etiqueta"] = "manual_500"

    desarrollo = pd.concat([manual_df, resto_df], ignore_index=True, sort=False)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    desarrollo.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"Dataset 80% etiquetado generado: {output_file}")
    print(desarrollo["sentimiento_final"].value_counts().to_string())


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Etiqueta automaticamente el resto del 80% de desarrollo.")
    parser.add_argument("--model-file", type=Path, default=MODEL_FILE)
    parser.add_argument("--resto", type=Path, default=RESTO_FILE)
    parser.add_argument("--manual", type=Path, default=MANUAL_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    etiquetar(args.model_file, args.resto, args.manual, args.output)


if __name__ == "__main__":
    main()
