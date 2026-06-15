"""Tests de los mapeos y helpers de etiquetas."""

from _comun.etiquetas import MAPEO_ESTRELLAS, etiqueta_por_estrellas, normalizar_estrellas


def test_mapeo_estrellas_extremos():
    assert MAPEO_ESTRELLAS[1] == "muy negativo"
    assert MAPEO_ESTRELLAS[3] == "neutral"
    assert MAPEO_ESTRELLAS[5] == "muy positivo"


def test_etiqueta_por_estrellas_valida():
    assert etiqueta_por_estrellas(3) == "neutral"
    assert etiqueta_por_estrellas("5") == "muy positivo"


def test_etiqueta_por_estrellas_invalida():
    assert etiqueta_por_estrellas(0) is None
    assert etiqueta_por_estrellas(6) is None
    assert etiqueta_por_estrellas("texto") is None
    assert etiqueta_por_estrellas("") is None


def test_normalizar_estrellas():
    assert normalizar_estrellas("4") == 4
    assert normalizar_estrellas(4.0) == 4
    assert normalizar_estrellas(7) is None
    assert normalizar_estrellas("abc") is None
