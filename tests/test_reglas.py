"""Tests de las reglas de etiquetado (fase 05b)."""

import pandas as pd

import etiquetar_por_reglas as reglas


def fila(estrellas, modelo, prob_neg=0.1, prob_neu=0.1, prob_pos=0.1):
    return pd.DataFrame(
        [{
            "estrellas": estrellas,
            "sentimiento_modelo": modelo,
            "prob_neg": prob_neg,
            "prob_neu": prob_neu,
            "prob_pos": prob_pos,
        }]
    )


def predecir(df):
    return reglas.predecir_por_reglas(df, umbral_neu=0.60, umbral_pos=0.25)


def test_r1_estrellas_altas_neu_confiado_da_neutral():
    etiqueta, regla = predecir(fila(5, "NEU", prob_neu=0.70))
    assert etiqueta.iloc[0] == "neutral"
    assert regla.iloc[0] == "R1_estrellas_altas_neu_neutral"


def test_r1_no_aplica_si_prob_neu_baja():
    etiqueta, _ = predecir(fila(4, "NEU", prob_neu=0.50))
    assert etiqueta.iloc[0] == ""


def test_r2_estrella_1_da_muy_negativo():
    etiqueta, regla = predecir(fila(1, "NEU", prob_pos=0.10))
    assert etiqueta.iloc[0] == "muy negativo"
    assert regla.iloc[0] == "R2_estrella_1_neu"


def test_r2_estrella_2_da_negativo():
    etiqueta, _ = predecir(fila(2, "NEU", prob_pos=0.10))
    assert etiqueta.iloc[0] == "negativo"


def test_r2_no_aplica_si_prob_pos_alta():
    etiqueta, _ = predecir(fila(1, "NEU", prob_pos=0.40))
    assert etiqueta.iloc[0] == ""


def test_contradiccion_dura_no_se_etiqueta():
    # 5 estrellas pero el modelo dice NEG: ambiguedad real, se deja sin etiqueta.
    etiqueta, _ = predecir(fila(5, "NEG", prob_neg=0.80))
    assert etiqueta.iloc[0] == ""


def test_tres_estrellas_residual_no_se_etiqueta():
    etiqueta, _ = predecir(fila(3, "NEG", prob_neg=0.60))
    assert etiqueta.iloc[0] == ""
