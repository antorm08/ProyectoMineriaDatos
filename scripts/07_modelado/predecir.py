"""Predice sentimiento para reseñas nuevas usando el mejor modelo entrenado.

Uso rapido:
    python scripts/07_modelado/predecir.py --texto "Muy buena atencion"
    python scripts/07_modelado/predecir.py --texto "Malo" --texto "Excelente servicio"
    python scripts/07_modelado/predecir.py --archivo nuevas_resenas.csv --columna comentario --salida predicciones.csv

Si no se pasa texto ni archivo, entra en modo interactivo.
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELO_FILE = PROJECT_ROOT / "models" / "mejor_modelo.joblib"

sys.path.append(str(PROJECT_ROOT / "scripts"))
from _comun.texto import texto_para_modelo  # noqa: E402


def cargar_modelo(ruta_modelo):
    """Carga el modelo entrenado desde disco.

    El archivo .joblib guarda dos piezas necesarias para predecir:
    - vectorizador: convierte texto limpio a variables TF-IDF.
    - clasificador: asigna la clase de sentimiento.
    """
    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No existe el modelo: {ruta_modelo}. Corre primero: python scripts/07_modelado/entrenar_evaluar.py"
        )

    # El warning aparece si scikit-learn tiene una version distinta a la usada al entrenar.
    # Se oculta para que el modo interactivo sea limpio; reentrenar elimina el warning de raiz.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        modelo = joblib.load(ruta_modelo)

    # Validacion minima: evita seguir si el archivo no tiene la estructura esperada.
    faltantes = {"vectorizador", "clasificador"} - set(modelo)
    if faltantes:
        raise ValueError(f"El modelo no tiene las claves esperadas: {sorted(faltantes)}")
    return modelo


def predecir(modelo, textos):
    """Convierte una o varias reseñas en predicciones de sentimiento.

    Flujo:
    1. Limpia cada reseña igual que durante el entrenamiento.
    2. Convierte el texto a TF-IDF con el vectorizador guardado.
    3. Usa el clasificador guardado para predecir el sentimiento.
    4. Si el clasificador entrega probabilidades, calcula la confianza.
    """
    # Misma limpieza usada para crear la columna texto_modelo en entrenamiento.
    textos_modelo = [texto_para_modelo(texto) for texto in textos]
    vectorizador = modelo["vectorizador"]
    clasificador = modelo["clasificador"]

    # El modelo no entiende texto crudo; primero se transforma a numeros TF-IDF.
    x = vectorizador.transform(textos_modelo)
    predicciones = clasificador.predict(x)

    # Regresion Logistica tiene predict_proba, por eso podemos mostrar confianza.
    probabilidades = None
    if hasattr(clasificador, "predict_proba"):
        probabilidades = clasificador.predict_proba(x)

    filas = []
    clases_prob = list(getattr(clasificador, "classes_", []))
    for i, (texto, texto_proc, prediccion) in enumerate(zip(textos, textos_modelo, predicciones)):
        fila = {
            "resena": texto,
            "texto_modelo": texto_proc,
            "sentimiento_predicho": prediccion,
        }
        if probabilidades is not None:
            probs = dict(zip(clases_prob, probabilidades[i]))
            # Confianza = probabilidad mas alta entre todas las clases.
            fila["confianza"] = round(float(max(probs.values())), 4)
            for clase in modelo.get("clases", clases_prob):
                if clase in probs:
                    fila[f"prob_{clase.replace(' ', '_')}"] = round(float(probs[clase]), 4)
        filas.append(fila)

    return pd.DataFrame(filas)


def predecir_desde_archivo(modelo, archivo, columna, salida):
    """Lee un CSV, predice la columna indicada y opcionalmente guarda otro CSV."""
    df = pd.read_csv(archivo)
    if columna not in df.columns:
        raise ValueError(f"El archivo no tiene la columna '{columna}'. Columnas disponibles: {list(df.columns)}")

    # Se predice cada comentario del CSV y se agregan las columnas de salida al archivo original.
    resultados = predecir(modelo, df[columna].fillna("").astype(str).tolist())
    salida_df = pd.concat([df.reset_index(drop=True), resultados.drop(columns=["resena"])], axis=1)

    if salida:
        salida_df.to_csv(salida, index=False, encoding="utf-8-sig")
        print(f"Predicciones guardadas en: {salida}")
    else:
        print(salida_df.to_string(index=False))


def modo_interactivo(modelo):
    """Permite escribir reseñas una por una desde la consola.

    Este modo se usa cuando no se pasan argumentos. Es util para probar rapidamente
    el modelo sin preparar un archivo CSV.
    """
    print("Modo interactivo. Escribe una reseña y presiona Enter. Escribe 'salir' para terminar.")
    while True:
        texto = input("> ").strip()

        # Palabras de salida para terminar el programa sin cerrar la terminal.
        if texto.lower() in {"salir", "exit", "quit"}:
            break

        # Si el usuario presiona Enter sin escribir nada, se vuelve a pedir otra reseña.
        if not texto:
            continue

        # Se reutiliza la funcion general de prediccion con una sola reseña.
        resultado = predecir(modelo, [texto]).iloc[0]
        confianza = f" | confianza: {resultado['confianza']:.4f}" if "confianza" in resultado else ""
        print(f"sentimiento: {resultado['sentimiento_predicho']}{confianza}")


def obtener_argumentos():
    """Define las opciones de ejecucion por consola.

    Si se usa --texto, predice textos escritos como argumento.
    Si se usa --archivo, predice reseñas desde CSV.
    Si no se usa ninguno, main activa el modo interactivo.
    """
    parser = argparse.ArgumentParser(description="Predice sentimiento para reseñas nuevas.")
    parser.add_argument("--texto", action="append", help="Reseña a predecir. Puede repetirse varias veces.")
    parser.add_argument("--archivo", type=Path, help="CSV con reseñas nuevas.")
    parser.add_argument("--columna", default="comentario", help="Columna del CSV que contiene la reseña.")
    parser.add_argument("--salida", type=Path, help="CSV donde guardar las predicciones.")
    parser.add_argument("--modelo", type=Path, default=MODELO_FILE, help="Ruta del modelo .joblib.")
    return parser.parse_args()


def main():
    """Punto de entrada: decide si usar texto, CSV o modo interactivo."""
    args = obtener_argumentos()
    modelo = cargar_modelo(args.modelo)

    # Modo lote: predice todas las reseñas de un CSV.
    if args.archivo:
        predecir_desde_archivo(modelo, args.archivo, args.columna, args.salida)
        return

    # Modo argumento: predice uno o varios --texto pasados por consola.
    if args.texto:
        resultados = predecir(modelo, args.texto)
        print(resultados.to_string(index=False))
        return

    # Sin argumentos: pide reseñas directamente en la consola.
    modo_interactivo(modelo)


if __name__ == "__main__":
    main()
