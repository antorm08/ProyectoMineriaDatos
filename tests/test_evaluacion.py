"""Tests del modulo comun de evaluacion (scripts/_comun/evaluacion.py).

Verifica la logica de metricas que comparten las tres familias de modelos (clasicos,
deep learning y transformers), para garantizar que se midan exactamente igual.
"""

from _comun.evaluacion import CLASES, f1_por_clase, metricas_basicas


def test_prediccion_perfecta_da_metricas_uno():
    y = ["muy negativo", "negativo", "neutral", "positivo", "muy positivo"]
    metricas = metricas_basicas(y, y)
    assert metricas["f1_macro"] == 1.0
    assert metricas["accuracy"] == 1.0
    assert metricas["balanced_accuracy"] == 1.0
    assert metricas["f1_weighted"] == 1.0


def test_f1_por_clase_respeta_orden_de_clases():
    y_true = ["muy negativo", "negativo", "neutral", "positivo", "muy positivo"]
    pares = f1_por_clase(y_true, y_true)
    assert [clase for clase, _ in pares] == CLASES
    assert all(valor == 1.0 for _, valor in pares)


def test_metricas_estan_acotadas_entre_cero_y_uno():
    y_true = ["muy negativo", "negativo", "neutral", "positivo", "muy positivo"]
    # Todo predicho como la clase mayoritaria: F1-macro debe ser bajo pero valido.
    y_pred = ["neutral"] * len(y_true)
    metricas = metricas_basicas(y_true, y_pred)
    for valor in metricas.values():
        assert 0.0 <= valor <= 1.0
    assert metricas["f1_macro"] < metricas["accuracy"] or metricas["f1_macro"] <= 0.5
