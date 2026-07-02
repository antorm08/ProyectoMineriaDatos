"""Fase 11: particion final 80/20 y seleccion de la semilla de 500 registros.

Primer paso del flujo semisupervisado indicado por el docente:

    1. El dataset completo se trata como NO etiquetado de forma confiable.
    2. Se separa un 20% como conjunto de prueba final, que se evalua UNA sola vez
       al final del flujo (fase 15).
    3. Del 80% de desarrollo se seleccionan 500 registros representativos que
       formaran la semilla etiquetada (fase 12) para la validacion cruzada (fase 13).

Como las etiquetas aun no existen en este punto del flujo, ambas selecciones se
estratifican por las senales DISPONIBLES en los datos crudos:

    - estrellas (1-5): senal debil de intensidad de sentimiento
    - terciles de longitud del comentario: representa resenas cortas/medias/largas

La estratificacion cruzada estrellas x longitud produce una semilla que cubre todo
el espectro de intensidad y de verbosidad, criterio defendible de representatividad.
Las etiquetas del pipeline anterior (sentimiento_final) se CONSERVAN como columna de
referencia para medir acuerdo, pero no participan en la seleccion.

Entrada:
    data/processed/dataset_consumidores_peru_etiquetado_final.csv

Salidas:
    data/splits_v2/test.csv             (20% prueba final, no tocar hasta fase 15)
    data/splits_v2/semilla.csv          (500 representativos del 80%)
    data/splits_v2/dev_resto.csv        (resto del 80%, a etiquetar por self-training)
    reports/10_particion_semilla/reporte_particion.csv
    reports/10_particion_semilla/distribucion_estrellas.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado_final.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits_v2"
REPORT_DIR = PROJECT_ROOT / "reports" / "10_particion_semilla"


def cargar_base():
    df = pd.read_csv(INPUT_FILE).fillna("")
    df = df[df["texto_modelo"].astype(str).str.strip() != ""].copy()
    df = df.reset_index().rename(columns={"index": "id_registro"})
    df["estrellas"] = pd.to_numeric(df["estrellas"], errors="coerce").fillna(3).astype(int)
    # Terciles de longitud: resenas cortas / medias / largas.
    df["tercil_longitud"] = pd.qcut(
        pd.to_numeric(df["cantidad_palabras"], errors="coerce").fillna(0),
        q=3, labels=["corta", "media", "larga"], duplicates="drop",
    ).astype(str)
    df["estrato"] = df["estrellas"].astype(str) + "|" + df["tercil_longitud"]
    return df


def distribucion(df, bloque):
    dist = df["estrellas"].value_counts().sort_index()
    return pd.DataFrame({
        "bloque": bloque,
        "estrellas": dist.index,
        "cantidad": dist.values,
        "porcentaje": (dist.values / len(df) * 100).round(2),
    })


def main():
    parser = argparse.ArgumentParser(description="Particion 80/20 y semilla de 500 (fase 11).")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--semilla-n", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = cargar_base()
    print(f"Base utilizable (texto no vacio): {len(df)} filas")

    # 20% de prueba final, estratificado por estrellas x longitud.
    dev_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state,
        stratify=df["estrato"],
    )

    # Semilla de 500 representativos dentro del 80% de desarrollo.
    semilla_df, resto_df = train_test_split(
        dev_df, train_size=args.semilla_n, random_state=args.random_state,
        stratify=dev_df["estrato"],
    )

    for nombre, bloque in [("test", test_df), ("semilla", semilla_df), ("dev_resto", resto_df)]:
        bloque.drop(columns=["estrato"]).to_csv(
            SPLITS_DIR / f"{nombre}.csv", index=False, encoding="utf-8-sig"
        )

    reporte = pd.DataFrame(
        [
            ("filas_base_utilizable", len(df)),
            ("filas_test", len(test_df)),
            ("filas_semilla", len(semilla_df)),
            ("filas_dev_resto", len(resto_df)),
            ("proporcion_test", round(len(test_df) / len(df), 4)),
            ("test_size", args.test_size),
            ("semilla_n", args.semilla_n),
            ("estratificacion", "estrellas x tercil_longitud"),
            ("random_state", args.random_state),
        ],
        columns=["metrica", "valor"],
    )
    reporte.to_csv(REPORT_DIR / "reporte_particion.csv", index=False, encoding="utf-8-sig")

    dist = pd.concat([
        distribucion(test_df, "test"),
        distribucion(semilla_df, "semilla"),
        distribucion(resto_df, "dev_resto"),
    ])
    dist.to_csv(REPORT_DIR / "distribucion_estrellas.csv", index=False, encoding="utf-8-sig")

    print(f"test:      {len(test_df)} filas -> {SPLITS_DIR / 'test.csv'}")
    print(f"semilla:   {len(semilla_df)} filas -> {SPLITS_DIR / 'semilla.csv'}")
    print(f"dev_resto: {len(resto_df)} filas -> {SPLITS_DIR / 'dev_resto.csv'}")
    print("\nDistribucion por estrellas (porcentaje):")
    tabla = dist.pivot(index="estrellas", columns="bloque", values="porcentaje")
    print(tabla.to_string())
    print(f"\nReportes en: {REPORT_DIR}")


if __name__ == "__main__":
    main()
