# Modelos entrenados

Los modelos ligeros se versionan aquí; los pesados (transformers y redes) se
distribuyen como **GitHub Release** porque superan el límite de 100 MB por archivo.

## Modelos versionados en el repo

| Archivo | Modelo | Tamaño |
|---|---|---|
| `mejor_modelo.joblib` | Mejor clásico del flujo por familias (SVM/NB + TF-IDF) | ~0.8 MB |
| `mejor_modelo_dl.pt` | Mejor red del flujo por familias (CNN/LSTM) | ~2.5 MB |

## Modelo principal (BETO) — descargar del Release

El mejor modelo del **flujo semisupervisado** (F1-macro test **0.6688**) es un BETO
fine-tuneado de ~420 MB. Se distribuye como release:

**Descarga:** https://github.com/antorm08/ProyectoMineriaDatos/releases/tag/modelo-beto-v2

```bash
# con GitHub CLI
gh release download modelo-beto-v2 --pattern "v2_mejor_modelo_beto.zip"
# descomprimir dentro de models/  ->  models/v2_mejor_modelo_transformer/
```

Uso:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
ruta = "models/v2_mejor_modelo_transformer"
tok = AutoTokenizer.from_pretrained(ruta)
modelo = AutoModelForSequenceClassification.from_pretrained(ruta)
```

Ambos modelos pesados se pueden regenerar corriendo el pipeline
(`scripts/15_entrenamiento_final/entrenar_final.py`).
