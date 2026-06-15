"""Tests de las utilidades de normalizacion de texto."""

from _comun.texto import (
    colapsar_espacios,
    contar_palabras,
    normalizar_comparacion,
    normalizar_texto,
    texto_para_modelo,
)


def test_normalizar_texto_colapsa_espacios_saltos_y_tabs():
    assert normalizar_texto("  hola\n\tmundo  ") == "hola mundo"


def test_normalizar_texto_quita_invisibles():
    entrada = "a" + chr(0x200B) + "b" + chr(0xFEFF) + "c"
    assert normalizar_texto(entrada) == "a b c"


def test_normalizar_texto_conserva_tildes_y_enie():
    assert normalizar_texto("  Café  Ñoño ") == "Café Ñoño"


def test_normalizar_texto_none_devuelve_vacio():
    assert normalizar_texto(None) == ""


def test_colapsar_espacios():
    assert colapsar_espacios("  a   b ") == "a b"


def test_texto_para_modelo_quita_url_signos_y_emojis():
    assert texto_para_modelo("¡Hola, MUNDO! http://x.com 😡") == "hola mundo"


def test_texto_para_modelo_conserva_acentos_y_numeros():
    assert texto_para_modelo("Café Ñoño 5★") == "café ñoño 5"


def test_normalizar_comparacion_minusculas_y_recorte():
    assert normalizar_comparacion("  SÍ  ") == "sí"


def test_contar_palabras():
    assert contar_palabras("hola, mundo cruel") == 3
    assert contar_palabras("") == 0
