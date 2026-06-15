"""Etiquetado automatico por reglas (cero IA, determinista y gratis).

Rellena `sentimiento_final` en las filas que quedaron sin etiqueta tras el consenso
estrellas+modelo y la revision IA, usando solo columnas ya calculadas
(`estrellas`, `sentimiento_modelo`, `prob_neg`, `prob_pos`). No ejecuta ningun
modelo: solo lee el CSV. Depende unicamente de pandas.

Reglas conservadoras (solo sobre filas con sentimiento_final vacio):
- R1: 4/5 estrellas + modelo NEU con prob_neu >= umbral_neu -> NEUTRAL.
      Validado empiricamente: cuando hay muchas estrellas pero el modelo dice NEU con
      confianza, la resena es genuinamente neutral (cliente generoso con la estrella,
      texto con peros), no positiva. 40/40 de estos casos ya revisados por humanos
      fueron neutral. Ademas refuerza la clase minoritaria neutral.
- R2: 1 estrella + modelo NEU + prob_pos < umbral_pos -> muy negativo.
      2 estrellas + modelo NEU + prob_pos < umbral_pos -> negativo.

No se tocan las contradicciones duras (estrellas altas + NEG, estrellas bajas + POS,
3 estrellas residual): se dejan sin etiqueta a proposito.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _comun.etiquetas import MAPEO_ESTRELLAS  # noqa: E402

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado_final.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado_final.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "03b_reglas"
REPORT_FILE = REPORT_DIR / "reporte_etiquetado_reglas.csv"
DISTRIBUTION_FILE = REPORT_DIR / "distribucion_final_reglas.csv"
VALIDATION_FILE = REPORT_DIR / "validacion_reglas.csv"

COLUMNAS_REQUERIDAS = {
    "estrellas",
    "sentimiento_modelo",
    "prob_neg",
    "prob_pos",
    "sentimiento_final",
}

ORIGEN_REGLA = "regla_voto_ponderado"
ORIGENES_IA = {"ia_clase_minoritaria", "ia_lote_neutral"}


def preparar_columnas(df):
    """Normaliza los tipos de las columnas que usan las reglas."""
    estrellas = pd.to_numeric(df["estrellas"], errors="coerce")
    # Probabilidades faltantes se tratan como 1.0: asi las condiciones "prob < umbral"
    # fallan de forma segura y la regla no asigna nada sobre datos incompletos.
    prob_neg = pd.to_numeric(df["prob_neg"], errors="coerce").fillna(1.0)
    prob_pos = pd.to_numeric(df["prob_pos"], errors="coerce").fillna(1.0)
    modelo = df["sentimiento_modelo"].astype(str).str.strip().str.upper()
    return estrellas, prob_neg, prob_pos, modelo


def predecir_por_reglas(df, umbral_neu, umbral_pos):
    """Devuelve dos Series: etiqueta sugerida por regla y nombre de la regla.

    La prediccion se calcula para TODAS las filas (ignorando si estan vacias o no),
    de modo que pueda reutilizarse tanto para asignar como para validar contra
    etiquetas IA existentes.
    """
    estrellas, _, prob_pos, modelo = preparar_columnas(df)
    prob_neu = pd.to_numeric(df["prob_neu"], errors="coerce").fillna(0.0)

    etiqueta = pd.Series("", index=df.index, dtype="object")
    regla = pd.Series("", index=df.index, dtype="object")

    es_neu = modelo == "NEU"

    # R1: estrellas altas con modelo confiadamente neutral -> neutral (no positivo).
    r1 = es_neu & estrellas.isin([4, 5]) & (prob_neu >= umbral_neu)
    etiqueta.loc[r1] = "neutral"
    regla.loc[r1] = "R1_estrellas_altas_neu_neutral"

    # R2: estrellas muy bajas con modelo tibio-neutral -> clase negativa por estrellas.
    r2_muy_neg = es_neu & (estrellas == 1) & (prob_pos < umbral_pos)
    etiqueta.loc[r2_muy_neg] = "muy negativo"
    regla.loc[r2_muy_neg] = "R2_estrella_1_neu"

    r2_neg = es_neu & (estrellas == 2) & (prob_pos < umbral_pos)
    etiqueta.loc[r2_neg] = "negativo"
    regla.loc[r2_neg] = "R2_estrella_2_neu"

    return etiqueta, regla


def construir_validacion(df, etiqueta_regla):
    """Compara lo que la regla predeciria contra las etiquetas IA ya existentes.

    Mide el % de acuerdo en las filas que (a) ya tienen una etiqueta hecha por IA y
    (b) la regla tambien podria etiquetar. Sirve para estimar la fiabilidad de las
    reglas antes de confiar en ellas sobre las filas vacias.
    """
    if "sentimiento_final_origen" not in df.columns:
        return None, pd.DataFrame()

    origen = df["sentimiento_final_origen"].astype(str).str.strip()
    es_ia = origen.isin(ORIGENES_IA)
    regla_aplica = etiqueta_regla.astype(str).str.strip() != ""
    mascara = es_ia & regla_aplica

    comparables = df.loc[mascara].copy()
    if comparables.empty:
        return 0, pd.DataFrame(columns=["sentimiento_ia", "etiqueta_regla", "cantidad"])

    comparables["etiqueta_regla"] = etiqueta_regla.loc[mascara]
    real = comparables["sentimiento_final"].astype(str).str.strip()
    acuerdo = (real == comparables["etiqueta_regla"]).mean()

    matriz = (
        comparables.assign(sentimiento_ia=real)
        .groupby(["sentimiento_ia", "etiqueta_regla"])
        .size()
        .reset_index(name="cantidad")
        .sort_values("cantidad", ascending=False)
    )
    return acuerdo, matriz


def etiquetar(input_file, output_file, umbral_neu, umbral_pos, dry_run):
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el dataset final: {input_file}")

    df = pd.read_csv(input_file).fillna("")
    faltantes = COLUMNAS_REQUERIDAS - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    if "sentimiento_final_origen" not in df.columns:
        df["sentimiento_final_origen"] = df["sentimiento_final"].apply(
            lambda valor: "consistencia_estrella_modelo" if str(valor).strip() else "sin_etiqueta"
        )

    final = df["sentimiento_final"].astype(str).str.strip()
    vacias_antes = int((final == "").sum())
    etiquetadas_antes = int((final != "").sum())

    etiqueta_regla, regla_aplicada = predecir_por_reglas(df, umbral_neu, umbral_pos)

    # Validacion contra etiquetas IA existentes (no depende de dry_run).
    acuerdo, matriz_validacion = construir_validacion(df, etiqueta_regla)

    # Solo asignamos sobre filas vacias.
    mascara_asignar = (final == "") & (etiqueta_regla.astype(str).str.strip() != "")
    indices = df.index[mascara_asignar]

    df_resultado = df.copy()
    df_resultado.loc[indices, "sentimiento_final"] = etiqueta_regla.loc[indices]
    df_resultado.loc[indices, "sentimiento_final_origen"] = ORIGEN_REGLA
    if "regla_aplicada" not in df_resultado.columns:
        df_resultado["regla_aplicada"] = ""
    df_resultado.loc[indices, "regla_aplicada"] = regla_aplicada.loc[indices]

    nuevas = int(len(indices))
    final_despues = df_resultado["sentimiento_final"].astype(str).str.strip()
    etiquetadas_despues = int((final_despues != "").sum())
    vacias_despues = int((final_despues == "").sum())

    conteo_por_regla = regla_aplicada.loc[indices].value_counts()

    distribucion = (
        final_despues[final_despues != ""]
        .value_counts()
        .rename_axis("sentimiento_final")
        .reset_index(name="cantidad")
    )

    reporte = pd.DataFrame(
        [
            ("filas_totales", len(df_resultado)),
            ("etiquetadas_antes", etiquetadas_antes),
            ("vacias_antes", vacias_antes),
            ("nuevas_por_regla", nuevas),
            ("nuevas_R1_estrellas_altas_neu_neutral", int(conteo_por_regla.get("R1_estrellas_altas_neu_neutral", 0))),
            ("nuevas_R2_estrella_1_neu", int(conteo_por_regla.get("R2_estrella_1_neu", 0))),
            ("nuevas_R2_estrella_2_neu", int(conteo_por_regla.get("R2_estrella_2_neu", 0))),
            ("etiquetadas_despues", etiquetadas_despues),
            ("residuo_sin_etiqueta", vacias_despues),
            ("umbral_neu", umbral_neu),
            ("umbral_pos", umbral_pos),
            ("acuerdo_validacion_ia", round(acuerdo, 4) if acuerdo is not None else "sin_datos"),
        ],
        columns=["metrica", "valor"],
    )

    # Salida.
    if dry_run:
        print("=== DRY-RUN: no se escribe ningun archivo ===")
    else:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        df_resultado.to_csv(output_file, index=False, encoding="utf-8-sig")
        reporte.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
        distribucion.to_csv(DISTRIBUTION_FILE, index=False, encoding="utf-8-sig")
        if matriz_validacion is not None:
            matriz_validacion.to_csv(VALIDATION_FILE, index=False, encoding="utf-8-sig")
        print(f"Dataset actualizado: {output_file}")
        print(f"Reporte: {REPORT_FILE}")
        print(f"Distribucion: {DISTRIBUTION_FILE}")
        print(f"Validacion: {VALIDATION_FILE}")

    print(f"\nArchivo entrada: {input_file}")
    print(f"Filas totales: {len(df_resultado)}")
    print(f"Etiquetadas antes: {etiquetadas_antes} | vacias antes: {vacias_antes}")
    print(f"Nuevas por regla: {nuevas}")
    if not conteo_por_regla.empty:
        print("\nRecuperadas por regla:")
        print(conteo_por_regla.to_string())
    print(f"\nEtiquetadas despues: {etiquetadas_despues} | residuo sin etiqueta: {vacias_despues}")
    print("\nValidacion contra etiquetas IA existentes:")
    if acuerdo is None:
        print("No hay columna de origen para validar.")
    elif matriz_validacion.empty:
        print("No hay filas IA comparables con las reglas.")
    else:
        n_comparables = int(matriz_validacion["cantidad"].sum())
        print(f"Acuerdo regla vs IA: {acuerdo:.1%} sobre {n_comparables} filas comparables.")
        print(matriz_validacion.to_string(index=False))
    print("\nDistribucion sentimiento_final tras reglas:")
    print(distribucion.to_string(index=False))


def obtener_argumentos():
    parser = argparse.ArgumentParser(
        description="Etiqueta por reglas conservadoras las filas sin sentimiento_final."
    )
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument(
        "--umbral-neu",
        type=float,
        default=0.60,
        help="R1 solo aplica si prob_neu es mayor o igual a este valor (estrellas altas + NEU -> neutral).",
    )
    parser.add_argument(
        "--umbral-pos",
        type=float,
        default=0.25,
        help="R2 solo aplica si prob_pos es menor a este valor (estrellas bajas + NEU).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No escribe archivos; solo imprime el resumen y la validacion.",
    )
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    etiquetar(args.input, args.output, args.umbral_neu, args.umbral_pos, args.dry_run)


if __name__ == "__main__":
    main()
