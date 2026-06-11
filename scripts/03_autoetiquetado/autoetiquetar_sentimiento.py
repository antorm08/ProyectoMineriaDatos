import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_limpio.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado.csv"
REPORT_FILE = PROJECT_ROOT / "reports" / "02_autoetiquetado" / "reporte_autoetiquetado.csv"
FINAL_DISTRIBUTION_FILE = PROJECT_ROOT / "reports" / "02_autoetiquetado" / "distribucion_sentimiento_final.csv"
PROVISIONAL_DISTRIBUTION_FILE = PROJECT_ROOT / "reports" / "02_autoetiquetado" / "distribucion_sentimiento_provisional.csv"
COLUMNAS_REQUERIDAS = {"comentario_limpio", "estrellas", "sentimiento_estrella"}

MAPEO_ESTRELLAS = {
    1: "muy negativo",
    2: "negativo",
    3: "neutral",
    4: "positivo",
    5: "muy positivo",
}

MAPEO_MODELO = {
    "NEG": "negativo",
    "NEU": "neutral",
    "POS": "positivo",
}


def cargar_analizador():
    try:
        from pysentimiento import create_analyzer
    except ImportError as error:
        raise SystemExit(
            "Falta instalar pysentimiento. Ejecuta: pip install -r requirements.txt"
        ) from error

    return create_analyzer(task="sentiment", lang="es")


def etiqueta_final_por_consistencia(estrellas, etiqueta_modelo):
    if pd.isna(estrellas):
        return None

    estrellas = int(estrellas)
    if estrellas not in MAPEO_ESTRELLAS:
        return None

    if etiqueta_modelo == "POS" and estrellas == 5:
        return "muy positivo"
    if etiqueta_modelo == "POS" and estrellas == 4:
        return "positivo"
    if etiqueta_modelo == "NEU" and estrellas == 3:
        return "neutral"
    if etiqueta_modelo == "NEG" and estrellas == 2:
        return "negativo"
    if etiqueta_modelo == "NEG" and estrellas == 1:
        return "muy negativo"

    return None


def etiqueta_provisional(estrellas, etiqueta_modelo):
    final = etiqueta_final_por_consistencia(estrellas, etiqueta_modelo)
    if final:
        return final

    if pd.isna(estrellas):
        return ""

    estrellas = int(estrellas)
    if estrellas not in MAPEO_ESTRELLAS:
        return ""

    return MAPEO_ESTRELLAS[estrellas]


def asignar_confianza_etiqueta(fila, umbral_confianza):
    if fila["requiere_revision_etiqueta"]:
        return "baja"
    if fila["confianza_modelo"] >= 0.80:
        return "alta"
    if fila["confianza_modelo"] >= umbral_confianza:
        return "media"
    return "baja"


def motivo_revision_etiqueta(fila, umbral_confianza):
    motivos = []

    if fila["sentimiento_final"] == "":
        motivos.append("sin_etiqueta_final_confiable")
    if fila["sentimiento_modelo"] == "SIN_TEXTO":
        motivos.append("comentario_vacio")
    if fila["confianza_modelo"] < umbral_confianza:
        motivos.append("baja_confianza_modelo")
    if fila.get("requiere_revision", False):
        motivos.append("revision_limpieza")

    return ";".join(motivos)


def autoetiquetar(
    input_file,
    output_file,
    report_file,
    distribution_file,
    provisional_distribution_file,
    umbral_confianza,
):
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el dataset limpio: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    distribution_file.parent.mkdir(parents=True, exist_ok=True)
    provisional_distribution_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file).fillna("")
    faltantes = COLUMNAS_REQUERIDAS - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    analizador = cargar_analizador()
    predicciones = []

    for comentario in df["comentario_limpio"].astype(str):
        comentario = comentario.strip()
        if comentario == "":
            predicciones.append(
                {
                    "sentimiento_modelo": "SIN_TEXTO",
                    "polaridad_modelo": "",
                    "confianza_modelo": 0.0,
                    "prob_neg": 0.0,
                    "prob_neu": 0.0,
                    "prob_pos": 0.0,
                }
            )
            continue

        resultado = analizador.predict(comentario)
        etiqueta = resultado.output
        probas = resultado.probas
        predicciones.append(
            {
                "sentimiento_modelo": etiqueta,
                "polaridad_modelo": MAPEO_MODELO.get(etiqueta, ""),
                "confianza_modelo": probas.get(etiqueta, 0.0),
                "prob_neg": probas.get("NEG", 0.0),
                "prob_neu": probas.get("NEU", 0.0),
                "prob_pos": probas.get("POS", 0.0),
            }
        )

    predicciones_df = pd.DataFrame(predicciones)
    df = pd.concat([df, predicciones_df], axis=1)
    df["estrellas"] = pd.to_numeric(df["estrellas"], errors="coerce")
    df["sentimiento_esperado"] = df["estrellas"].map(MAPEO_ESTRELLAS)
    df["sentimiento_final"] = df.apply(
        lambda fila: etiqueta_final_por_consistencia(fila["estrellas"], fila["sentimiento_modelo"]),
        axis=1,
    ).fillna("")

    df["sentimiento_final_provisional"] = df.apply(
        lambda fila: etiqueta_provisional(fila["estrellas"], fila["sentimiento_modelo"]),
        axis=1,
    )

    if "requiere_revision" in df.columns:
        revision_limpieza = df["requiere_revision"].astype(bool)
    else:
        revision_limpieza = pd.Series(False, index=df.index)

    df["requiere_revision_etiqueta"] = (
        (df["sentimiento_final"] == "")
        | (df["confianza_modelo"] < umbral_confianza)
        | revision_limpieza
    )
    df["confianza_etiqueta"] = df.apply(
        lambda fila: asignar_confianza_etiqueta(fila, umbral_confianza),
        axis=1,
    )
    df["motivo_revision_etiqueta"] = df.apply(
        lambda fila: motivo_revision_etiqueta(fila, umbral_confianza),
        axis=1,
    )

    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    reporte = pd.DataFrame(
        [
            ("filas", len(df)),
            ("etiquetas_finales_asignadas", int((df["sentimiento_final"] != "").sum())),
            ("etiquetas_provisionales_asignadas", int((df["sentimiento_final_provisional"] != "").sum())),
            ("filas_revision_etiqueta", int(df["requiere_revision_etiqueta"].sum())),
            ("confianza_alta", int((df["confianza_etiqueta"] == "alta").sum())),
            ("confianza_media", int((df["confianza_etiqueta"] == "media").sum())),
            ("confianza_baja", int((df["confianza_etiqueta"] == "baja").sum())),
            ("umbral_confianza", umbral_confianza),
        ],
        columns=["metrica", "valor"],
    )
    reporte.to_csv(report_file, index=False, encoding="utf-8-sig")

    distribucion = (
        df.loc[df["sentimiento_final"] != "", "sentimiento_final"]
        .value_counts()
        .rename_axis("sentimiento_final")
        .reset_index(name="cantidad")
    )
    distribucion.to_csv(distribution_file, index=False, encoding="utf-8-sig")

    distribucion_provisional = (
        df.loc[df["sentimiento_final_provisional"] != "", "sentimiento_final_provisional"]
        .value_counts()
        .rename_axis("sentimiento_final_provisional")
        .reset_index(name="cantidad")
    )
    distribucion_provisional.to_csv(
        provisional_distribution_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Archivo entrada: {input_file}")
    print(f"Archivo salida: {output_file}")
    print(f"Reporte salida: {report_file}")
    print(f"Distribucion final salida: {distribution_file}")
    print(f"Distribucion provisional salida: {provisional_distribution_file}")
    print(f"Filas: {len(df)}")
    print(f"Etiquetas finales asignadas: {(df['sentimiento_final'] != '').sum()}")
    print(f"Etiquetas provisionales asignadas: {(df['sentimiento_final_provisional'] != '').sum()}")
    print(f"Filas para revision de etiqueta: {df['requiere_revision_etiqueta'].sum()}")
    print("\nConfianza de etiqueta:")
    print(df["confianza_etiqueta"].value_counts().to_string())
    print("\nDistribucion sentimiento_modelo:")
    print(df["sentimiento_modelo"].value_counts().to_string())
    print("\nDistribucion sentimiento_final asignado:")
    print(distribucion.to_string(index=False))
    print("\nDistribucion sentimiento_final_provisional:")
    print(distribucion_provisional.to_string(index=False))


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Autoetiqueta sentimientos usando pysentimiento.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--reporte", type=Path, default=REPORT_FILE)
    parser.add_argument("--distribucion", type=Path, default=FINAL_DISTRIBUTION_FILE)
    parser.add_argument("--distribucion-provisional", type=Path, default=PROVISIONAL_DISTRIBUTION_FILE)
    parser.add_argument(
        "--umbral-confianza",
        type=float,
        default=0.60,
        help="Marca para revision predicciones con confianza menor a este valor.",
    )
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    autoetiquetar(
        args.input,
        args.output,
        args.reporte,
        args.distribucion,
        args.distribucion_provisional,
        args.umbral_confianza,
    )


if __name__ == "__main__":
    main()
