"""Helpers de DataFrame compartidos por las etapas de integracion.

Antes `validar_columnas` y la logica de mapeo de id_revision -> fila del dataset
estaban duplicadas en integrar_etiquetas_ia.py e integrar_etiquetas_ia_neutral.py.
"""

import pandas as pd


def validar_columnas(df, columnas, nombre):
    faltantes = [columna for columna in columnas if columna not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en {nombre}: {faltantes}")


def mapear_ia_a_dataset(dataset, base, ia, columnas_mapeo, columna_id, nombre_base, nombre_ia):
    """Asocia cada fila de etiquetas IA con el indice de fila del dataset original.

    1) Mapea el archivo `base` (revision) al dataset por `columnas_mapeo`.
    2) Mapea el archivo `ia` a `base` por `columna_id`.
    Devuelve `ia` con una columna `dataset_index`. Lanza ValueError si algun mapeo falla.
    """
    base = base.copy()
    ia = ia.copy()

    dataset_mapeo = dataset.reset_index().rename(columns={"index": "dataset_index"})
    if "estrellas" in columnas_mapeo:
        dataset_mapeo["estrellas"] = pd.to_numeric(dataset_mapeo["estrellas"], errors="coerce")
        base["estrellas"] = pd.to_numeric(base["estrellas"], errors="coerce")

    base[columna_id] = pd.to_numeric(base[columna_id], errors="coerce")
    ia[columna_id] = pd.to_numeric(ia[columna_id], errors="coerce")

    base = base.merge(dataset_mapeo[["dataset_index", *columnas_mapeo]], on=columnas_mapeo, how="left")
    if base["dataset_index"].isna().any():
        faltantes = base.loc[base["dataset_index"].isna(), columna_id].tolist()
        raise ValueError(f"No se pudo mapear {columna_id} al dataset desde {nombre_base}: {faltantes[:10]}")

    ia = ia.merge(base[[columna_id, "dataset_index"]], on=columna_id, how="left")
    if ia["dataset_index"].isna().any():
        faltantes = ia.loc[ia["dataset_index"].isna(), columna_id].tolist()
        raise ValueError(f"Hay {columna_id} en {nombre_ia} que no existen en {nombre_base}: {faltantes[:10]}")

    return ia
