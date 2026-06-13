"""Utilidades compartidas para el etiquetado de sentimiento.

Centraliza el mapeo de estrellas a clases y los helpers de polaridad que hoy
estan duplicados en varios scripts (limpiar_dataset.py, autoetiquetar_sentimiento.py,
integrar_etiquetas_ia*.py). El script de etiquetado por reglas los reutiliza desde aqui.
"""

import pandas as pd


# Etiqueta debil derivada de las estrellas de Google (1..5).
MAPEO_ESTRELLAS = {
    1: "muy negativo",
    2: "negativo",
    3: "neutral",
    4: "positivo",
    5: "muy positivo",
}

# Salida de pysentimiento (3 clases) a polaridad legible.
MAPEO_MODELO = {
    "NEG": "negativo",
    "NEU": "neutral",
    "POS": "positivo",
}

# Clases ordenadas de mas negativa a mas positiva (util para reportes/orden).
CLASES_ORDENADAS = [
    "muy negativo",
    "negativo",
    "neutral",
    "positivo",
    "muy positivo",
]


def normalizar_estrellas(valor):
    """Convierte el valor de estrellas a int en 1..5 o None si es invalido."""
    numero = pd.to_numeric(valor, errors="coerce")
    if pd.isna(numero):
        return None

    numero = int(numero)
    return numero if numero in MAPEO_ESTRELLAS else None


def etiqueta_por_estrellas(valor):
    """Devuelve la etiqueta debil esperada segun las estrellas, o None."""
    estrellas = normalizar_estrellas(valor)
    return MAPEO_ESTRELLAS.get(estrellas)
