import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "dataset_consumidores_peru.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_limpio.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "01_limpieza" / "reporte_limpieza_dataset.csv"
SENTIMENT_DISTRIBUTION_FILE = PROJECT_ROOT / "reports" / "01_limpieza" / "distribucion_sentimiento.csv"
STARS_DISTRIBUTION_FILE = PROJECT_ROOT / "reports" / "01_limpieza" / "distribucion_estrellas.csv"
MAPEO_ESTRELLAS = {
    1: "muy negativo",
    2: "negativo",
    3: "neutral",
    4: "positivo",
    5: "muy positivo",
}
COLUMNAS_REQUERIDAS = {
    "comentario",
    "empresa",
    "sede",
    "rubro",
    "estrellas",
    "sentimiento_estrella",
    "fecha_resena",
    "url",
}


def normalizar_texto(texto):
    # Mantiene tildes, ñ, signos y emojis; solo limpia caracteres invisibles y espacios.
    texto = "" if pd.isna(texto) else str(texto)
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace("\u200b", " ").replace("\ufeff", " ")
    texto = re.sub(r"[\r\n\t]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def contar_palabras(texto):
    return len(re.findall(r"\b\w+\b", texto, flags=re.UNICODE))


def texto_para_modelo(texto):
    # Version mas normalizada para TF-IDF/modelos clasicos, sin reemplazar el texto original.
    texto = normalizar_texto(texto).lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúñü0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def construir_motivo_revision(fila):
    # No elimina estos registros: deja trazabilidad para decidir si se revisan manualmente.
    motivos = []

    if fila["comentario_limpio"] == "":
        motivos.append("comentario_vacio")
    if not fila["estrellas_validas"]:
        motivos.append("estrellas_invalidas")
    if not fila["sentimiento_consistente"]:
        motivos.append("sentimiento_inconsistente_con_estrellas")
    if fila["comentario_corto"]:
        motivos.append("comentario_corto")
    if fila["sin_contenido_alfabetico"]:
        motivos.append("sin_contenido_alfabetico")
    if fila["duplicado_global"]:
        motivos.append("comentario_repetido_en_dataset")
    if fila["duplicado_empresa_sede"]:
        motivos.append("duplicado_misma_empresa_sede")

    return ";".join(motivos)


def guardar_reporte_limpieza(
    df_limpio,
    reporte_file,
    distribucion_sentimiento_file,
    distribucion_estrellas_file,
    filas_iniciales,
    duplicados_removidos,
    min_palabras,
):
    metricas = [
        ("filas_iniciales", filas_iniciales),
        ("filas_finales", len(df_limpio)),
        ("duplicados_exactos_removidos", int(duplicados_removidos)),
        ("comentarios_vacios", int((df_limpio["comentario_limpio"] == "").sum())),
        ("estrellas_invalidas", int((~df_limpio["estrellas_validas"]).sum())),
        ("sentimientos_inconsistentes", int((~df_limpio["sentimiento_consistente"]).sum())),
        (f"comentarios_cortos_menos_{min_palabras}_palabras", int(df_limpio["comentario_corto"].sum())),
        ("sin_contenido_alfabetico", int(df_limpio["sin_contenido_alfabetico"].sum())),
        ("filas_para_revision", int(df_limpio["requiere_revision"].sum())),
    ]
    reporte = pd.DataFrame(metricas, columns=["metrica", "valor"])
    reporte.to_csv(reporte_file, index=False, encoding="utf-8-sig")

    distribucion_sentimiento = (
        df_limpio["sentimiento_estrella"]
        .value_counts()
        .rename_axis("sentimiento_estrella")
        .reset_index(name="cantidad")
    )
    distribucion_sentimiento.to_csv(
        distribucion_sentimiento_file,
        index=False,
        encoding="utf-8-sig",
    )

    distribucion_estrellas = (
        df_limpio["estrellas"]
        .value_counts()
        .sort_index()
        .rename_axis("estrellas")
        .reset_index(name="cantidad")
    )
    distribucion_estrellas.to_csv(
        distribucion_estrellas_file,
        index=False,
        encoding="utf-8-sig",
    )


def limpiar_dataset(
    input_file,
    output_file,
    reporte_file,
    distribucion_sentimiento_file,
    distribucion_estrellas_file,
    min_palabras,
):
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el dataset: {input_file}")

    df = pd.read_csv(input_file).fillna("")
    faltantes = COLUMNAS_REQUERIDAS - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    filas_iniciales = len(df)

    # Normalizacion ligera de columnas textuales sin alterar el sentido del comentario.
    for columna in ["comentario", "empresa", "sede", "rubro", "fecha_resena", "url"]:
        df[columna] = df[columna].map(normalizar_texto)

    # Columnas de auditoria utiles para filtrar o justificar decisiones posteriores.
    df["comentario_limpio"] = df["comentario"].map(normalizar_texto)
    df["texto_modelo"] = df["comentario_limpio"].map(texto_para_modelo)
    df["longitud_caracteres"] = df["comentario_limpio"].str.len()
    df["cantidad_palabras"] = df["comentario_limpio"].map(contar_palabras)
    df["comentario_corto"] = df["cantidad_palabras"] < min_palabras
    df["sin_contenido_alfabetico"] = ~df["comentario_limpio"].str.contains(
        r"[a-zA-ZáéíóúÁÉÍÓÚñÑ]",
        regex=True,
    )
    df["duplicado_global"] = df.duplicated(subset=["comentario_limpio"], keep=False)
    df["duplicado_empresa_sede"] = df.duplicated(
        subset=["comentario_limpio", "empresa", "sede"],
        keep=False,
    )

    # Las estrellas se conservan como etiqueta debil inicial, pero se valida su rango.
    df["estrellas"] = pd.to_numeric(df["estrellas"], errors="coerce")
    df["estrellas_validas"] = df["estrellas"].isin([1, 2, 3, 4, 5])
    df["sentimiento_estrella"] = df["sentimiento_estrella"].map(normalizar_texto).str.lower().str.strip()
    df["sentimiento_esperado"] = df["estrellas"].map(MAPEO_ESTRELLAS)
    df["sentimiento_consistente"] = df["estrellas_validas"] & (
        df["sentimiento_estrella"] == df["sentimiento_esperado"]
    )
    df["requiere_revision"] = (
        (df["comentario_limpio"] == "")
        | ~df["estrellas_validas"]
        | ~df["sentimiento_consistente"]
        | df["comentario_corto"]
        | df["sin_contenido_alfabetico"]
        | df["duplicado_global"]
        | df["duplicado_empresa_sede"]
    )
    df["motivo_revision"] = df.apply(construir_motivo_revision, axis=1)

    # Solo se remueven duplicados exactos dentro de la misma empresa y sede.
    duplicados_exactos = df.duplicated(
        subset=["comentario_limpio", "empresa", "sede"],
        keep="first",
    )
    df_limpio = df[~duplicados_exactos].copy()
    df_limpio = df_limpio.sort_values(["empresa", "sede", "estrellas", "fecha_resena"])
    df_limpio.to_csv(output_file, index=False, encoding="utf-8-sig")
    guardar_reporte_limpieza(
        df_limpio,
        reporte_file,
        distribucion_sentimiento_file,
        distribucion_estrellas_file,
        filas_iniciales,
        duplicados_exactos.sum(),
        min_palabras,
    )

    # Resumen reproducible para documentar la etapa de limpieza.
    print(f"Archivo entrada: {input_file}")
    print(f"Archivo salida: {output_file}")
    print(f"Reporte salida: {reporte_file}")
    print(f"Distribucion sentimiento salida: {distribucion_sentimiento_file}")
    print(f"Distribucion estrellas salida: {distribucion_estrellas_file}")
    print(f"Filas iniciales: {filas_iniciales}")
    print(f"Duplicados exactos removidos: {duplicados_exactos.sum()}")
    print(f"Filas finales: {len(df_limpio)}")
    print(f"Comentarios vacios: {(df_limpio['comentario_limpio'] == '').sum()}")
    print(f"Estrellas invalidas: {(~df_limpio['estrellas_validas']).sum()}")
    print(f"Sentimientos inconsistentes: {(~df_limpio['sentimiento_consistente']).sum()}")
    print(f"Comentarios cortos (<{min_palabras} palabras): {df_limpio['comentario_corto'].sum()}")
    print(f"Sin contenido alfabetico: {df_limpio['sin_contenido_alfabetico'].sum()}")
    print(f"Filas para revision: {df_limpio['requiere_revision'].sum()}")
    print("\nDistribucion por sentimiento_estrella:")
    print(df_limpio["sentimiento_estrella"].value_counts().to_string())
    print("\nDistribucion por estrellas:")
    print(df_limpio["estrellas"].value_counts().sort_index().to_string())
    print("\nSedes con menos registros:")
    print(df_limpio.groupby(["empresa", "sede"]).size().sort_values().head(15).to_string())


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Limpia y audita el dataset de resenas.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--reporte", type=Path, default=REPORT_FILE)
    parser.add_argument("--distribucion-sentimiento", type=Path, default=SENTIMENT_DISTRIBUTION_FILE)
    parser.add_argument("--distribucion-estrellas", type=Path, default=STARS_DISTRIBUTION_FILE)
    parser.add_argument(
        "--min-palabras",
        type=int,
        default=3,
        help="Marca comentarios con menos de este numero de palabras para revision.",
    )
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    limpiar_dataset(
        args.input,
        args.output,
        args.reporte,
        args.distribucion_sentimiento,
        args.distribucion_estrellas,
        args.min_palabras,
    )


if __name__ == "__main__":
    main()
