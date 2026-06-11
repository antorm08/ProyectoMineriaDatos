import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_prioritaria_etiquetas.csv"

OPCIONES_SENTIMIENTO = {
    "1": "muy negativo",
    "2": "negativo",
    "3": "neutral",
    "4": "positivo",
    "5": "muy positivo",
}


def limpiar_pantalla():
    print("\n" * 3)


def mostrar_resena(fila, posicion, total):
    limpiar_pantalla()
    print(f"Revision {posicion}/{total} | id_revision: {fila.get('id_revision', '')}")
    print(f"Prioridad: {fila.get('prioridad_revision', '')} | Tipo: {fila.get('tipo_revision', '')}")
    print(f"Empresa: {fila.get('empresa', '')} | Sede: {fila.get('sede', '')} | Rubro: {fila.get('rubro', '')}")
    print("-" * 90)
    print(f"Comentario:\n{fila.get('comentario_limpio', fila.get('comentario', ''))}")
    print("-" * 90)
    print(f"Estrellas: {fila.get('estrellas', '')} | Etiqueta estrella: {fila.get('sentimiento_estrella', '')}")
    print(
        "Modelo: "
        f"{fila.get('sentimiento_modelo', '')} | "
        f"confianza: {float(fila.get('confianza_modelo', 0.0)):.3f} | "
        f"NEG: {float(fila.get('prob_neg', 0.0)):.3f} | "
        f"NEU: {float(fila.get('prob_neu', 0.0)):.3f} | "
        f"POS: {float(fila.get('prob_pos', 0.0)):.3f}"
    )
    print(f"Provisional: {fila.get('sentimiento_final_provisional', '')}")
    print(f"Motivo revision: {fila.get('motivo_revision_etiqueta', '')}")
    print("\nOpciones:")
    print("1 = muy negativo")
    print("2 = negativo")
    print("3 = neutral")
    print("4 = positivo")
    print("5 = muy positivo")
    print("s = saltar")
    print("q = guardar y salir")


def guardar(df, input_file):
    df.to_csv(input_file, index=False, encoding="utf-8-sig")


def revisar(input_file, limite):
    if not input_file.exists():
        raise FileNotFoundError(f"No existe el archivo de revision: {input_file}")

    df = pd.read_csv(input_file).fillna("")
    for columna in ["sentimiento_manual", "observacion_manual"]:
        if columna not in df.columns:
            df[columna] = ""

    pendientes = df[df["sentimiento_manual"].astype(str).str.strip() == ""].index.tolist()
    if limite > 0:
        pendientes = pendientes[:limite]

    if not pendientes:
        print("No hay filas pendientes de revision manual.")
        return

    total = len(pendientes)
    revisadas = 0

    for posicion, indice in enumerate(pendientes, start=1):
        fila = df.loc[indice]
        mostrar_resena(fila, posicion, total)
        respuesta = input("Elige una opcion: ").strip().lower()

        if respuesta == "q":
            guardar(df, input_file)
            print(f"Avance guardado en: {input_file}")
            print(f"Resenas marcadas en esta sesion: {revisadas}")
            return

        if respuesta == "s" or respuesta == "":
            continue

        if respuesta not in OPCIONES_SENTIMIENTO:
            print("Opcion invalida. Se salta esta resena.")
            continue

        df.at[indice, "sentimiento_manual"] = OPCIONES_SENTIMIENTO[respuesta]
        observacion = input("Observacion opcional (Enter para dejar vacio): ").strip()
        if observacion:
            df.at[indice, "observacion_manual"] = observacion

        revisadas += 1
        guardar(df, input_file)

    guardar(df, input_file)
    print(f"Revision finalizada. Avance guardado en: {input_file}")
    print(f"Resenas marcadas en esta sesion: {revisadas}")


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Revision manual interactiva de etiquetas.")
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument(
        "--limite",
        type=int,
        default=0,
        help="Cantidad maxima de pendientes a revisar en esta sesion. 0 revisa todas.",
    )
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    revisar(args.input, args.limite)


if __name__ == "__main__":
    main()
