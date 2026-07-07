"""Carga de los splits train/valid/test compartida por las fases de modelado.

Las fases 07 (clasicos), 08 (deep learning) y 09 (transformers) parten exactamente del
mismo dataset particionado en la fase 06. Centralizar la carga garantiza que las tres
familias entrenen y evaluen sobre los mismos textos y etiquetas, sin filas vacias.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

COLUMNA_TEXTO = "texto_modelo"
COLUMNA_ETIQUETA = "sentimiento_final"


def cargar_split(nombre, splits_dir=SPLITS_DIR, columna_texto=COLUMNA_TEXTO):
    """Carga un split y devuelve (textos, etiquetas) como Series de texto.

    El filtrado de filas SIEMPRE se hace sobre `texto_modelo` + etiqueta, de modo que el
    conjunto de filas es identico entre familias sin importar que columna de texto se pida.
    Asi los clasicos y las redes pueden usar `texto_modelo` (limpio, en minusculas) y los
    transformers `comentario_limpio` (texto natural, con mayusculas y signos) evaluando
    exactamente sobre las mismas resenas.
    """
    ruta = Path(splits_dir) / f"{nombre}.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe el split {nombre}: {ruta}. Corre primero la fase 06 (preparar_split_dataset.py)."
        )

    df = pd.read_csv(ruta).fillna("")
    faltantes = {COLUMNA_TEXTO, COLUMNA_ETIQUETA, columna_texto} - set(df.columns)
    if faltantes:
        raise ValueError(f"El split {nombre} no tiene columnas requeridas: {sorted(faltantes)}")

    df = df[
        (df[COLUMNA_TEXTO].astype(str).str.strip() != "")
        & (df[COLUMNA_ETIQUETA].astype(str).str.strip() != "")
    ]
    return df[columna_texto].astype(str), df[COLUMNA_ETIQUETA].astype(str)
