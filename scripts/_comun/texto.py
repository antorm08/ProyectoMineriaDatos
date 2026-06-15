"""Utilidades de normalizacion de texto compartidas.

Antes estas funciones estaban duplicadas (con semanticas distintas) en varios scripts:
- scriptscraping.py y auditar_dataset.py colapsaban espacios (colapsar_espacios).
- limpiar_dataset.py hacia normalizacion NFC + quita invisibles (normalizar_texto).
- los scripts de integracion comparaban en minusculas (normalizar_comparacion).

Centralizarlas evita que un mismo comentario se normalice distinto segun la etapa.
"""

import re
import unicodedata

import pandas as pd

ESPACIO_CERO = chr(0x200B)  # zero-width space
BOM = chr(0xFEFF)           # byte order mark


def colapsar_espacios(texto):
    """Colapsa espacios en blanco y recorta. Version ligera (scraping, auditoria)."""
    texto = "" if pd.isna(texto) else str(texto)
    return re.sub(r"\s+", " ", texto).strip()


def normalizar_texto(texto):
    """Normalizacion canonica del comentario (limpieza).

    Mantiene tildes, ñ, signos y emojis; solo limpia caracteres invisibles y espacios.
    """
    texto = "" if pd.isna(texto) else str(texto)
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace(ESPACIO_CERO, " ").replace(BOM, " ")
    texto = re.sub(r"[\r\n\t]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def contar_palabras(texto):
    return len(re.findall(r"\b\w+\b", texto, flags=re.UNICODE))


def texto_para_modelo(texto):
    """Version mas normalizada para TF-IDF/modelos clasicos, sin alterar el original."""
    texto = normalizar_texto(texto).lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúñü0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def normalizar_comparacion(texto):
    """Normaliza un valor para compararlo sin importar mayusculas (integracion)."""
    return "" if pd.isna(texto) else str(texto).strip().lower()
