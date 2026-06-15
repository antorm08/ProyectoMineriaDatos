"""Tests de la consolidacion de etiqueta por consenso estrellas+modelo (fase 03)."""

import autoetiquetar_sentimiento as auto


def test_consenso_asigna_etiqueta():
    assert auto.etiqueta_final_por_consistencia(5, "POS") == "muy positivo"
    assert auto.etiqueta_final_por_consistencia(4, "POS") == "positivo"
    assert auto.etiqueta_final_por_consistencia(3, "NEU") == "neutral"
    assert auto.etiqueta_final_por_consistencia(2, "NEG") == "negativo"
    assert auto.etiqueta_final_por_consistencia(1, "NEG") == "muy negativo"


def test_sin_consenso_no_asigna():
    assert auto.etiqueta_final_por_consistencia(5, "NEG") is None
    assert auto.etiqueta_final_por_consistencia(1, "POS") is None
    assert auto.etiqueta_final_por_consistencia(3, "POS") is None


def test_provisional_cae_a_estrellas_si_no_hay_consenso():
    assert auto.etiqueta_provisional(5, "NEG") == "muy positivo"
    assert auto.etiqueta_provisional(3, "POS") == "neutral"


def test_provisional_usa_consenso_si_existe():
    assert auto.etiqueta_provisional(5, "POS") == "muy positivo"
