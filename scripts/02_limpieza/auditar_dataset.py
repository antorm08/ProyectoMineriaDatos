import argparse
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = PROJECT_ROOT / "data" / "raw" / "dataset_consumidores_peru.csv"
EMPRESAS_FILE = PROJECT_ROOT / "data" / "raw" / "empresas.csv"
COLUMNAS_DATASET = {"comentario", "empresa", "sede", "estrellas", "url"}


def normalizar_texto(texto):
    return re.sub(r"\s+", " ", str(texto)).strip()


def cargar_csv(path, nombre):
    if not path.exists():
        raise FileNotFoundError(f"No existe {nombre}: {path}")

    return pd.read_csv(path).fillna("")


def buscar_menciones_otras_marcas(df, empresas_df):
    marcas = set(empresas_df["nombre"].astype(str).map(normalizar_texto))
    marcas.update(["Interbank", "BBVA", "Scotiabank", "BanBif", "Yape"])
    marcas = sorted([marca for marca in marcas if marca], key=len, reverse=True)

    sospechosas = []
    for _, fila in df.iterrows():
        comentario = normalizar_texto(fila["comentario"])
        comentario_lower = comentario.lower()
        empresa_lower = normalizar_texto(fila["empresa"]).lower()
        menciones = []

        for marca in marcas:
            marca_lower = marca.lower()
            if marca_lower in empresa_lower:
                continue

            patron = r"(?<![a-záéíóúñ])" + re.escape(marca_lower) + r"(?![a-záéíóúñ])"
            if re.search(patron, comentario_lower):
                menciones.append(marca)

        if menciones:
            sospechosas.append(
                {
                    "empresa": fila["empresa"],
                    "sede": fila["sede"],
                    "menciones": ", ".join(menciones),
                    "comentario": comentario[:220],
                }
            )

    return pd.DataFrame(sospechosas)


def auditar(dataset_file, empresas_file, max_reviews, ejemplos):
    df = cargar_csv(dataset_file, "dataset")
    empresas_df = cargar_csv(empresas_file, "empresas")

    faltantes = COLUMNAS_DATASET - set(df.columns)
    if faltantes:
        raise ValueError(f"El dataset no tiene estas columnas requeridas: {sorted(faltantes)}")

    df = df.copy()
    for columna in ["comentario", "empresa", "sede"]:
        df[columna] = df[columna].astype(str).map(normalizar_texto)

    duplicados = df.duplicated(subset=["comentario", "empresa", "sede"]).sum()
    sin_comentario = (df["comentario"] == "").sum()
    sin_estrellas = (df["estrellas"].astype(str).map(normalizar_texto) == "").sum()
    conteos = df.groupby(["empresa", "sede"]).size().sort_values()
    incompletas = conteos[conteos < max_reviews]
    menciones = buscar_menciones_otras_marcas(df, empresas_df)

    print(f"Filas dataset: {len(df)}")
    print(f"Sedes en dataset: {len(conteos)}")
    print(f"Filas empresas.csv: {len(empresas_df)}")
    print(f"Duplicados exactos comentario+empresa+sede: {duplicados}")
    print(f"Filas sin comentario: {sin_comentario}")
    print(f"Filas sin estrellas: {sin_estrellas}")

    print("\nSedes incompletas:")
    if incompletas.empty:
        print(f"Todas tienen al menos {max_reviews} resenas.")
    else:
        print(incompletas.to_string())

    print("\nMenciones de otras marcas:")
    if menciones.empty:
        print("No se encontraron menciones cruzadas.")
    else:
        resumen = menciones.groupby(["empresa", "sede"]).size().sort_values(ascending=False)
        print(resumen.head(20).to_string())
        print("\nEjemplos:")
        for _, fila in menciones.head(ejemplos).iterrows():
            print(f"- {fila['empresa']} | {fila['sede']} | {fila['menciones']} | {fila['comentario']}")


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Audita posibles problemas del dataset de resenas.")
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--empresas", type=Path, default=EMPRESAS_FILE)
    parser.add_argument("--max-reviews", type=int, default=150)
    parser.add_argument("--ejemplos", type=int, default=20)
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    auditar(args.dataset, args.empresas, args.max_reviews, args.ejemplos)


if __name__ == "__main__":
    main()
