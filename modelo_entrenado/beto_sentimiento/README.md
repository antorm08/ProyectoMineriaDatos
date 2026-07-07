# Modelo BETO entrenado — Clasificación de sentimiento (5 clases)

Mejor modelo del flujo semisupervisado (fases 11-15). Fine-tuning de **BETO**
(`dccuchile/bert-base-spanish-wwm-cased`).

> **Este modelo se versiona con Git LFS** (pesa ~420 MB). Para obtenerlo tras clonar:
> ```bash
> git lfs install
> git lfs pull
> ```

## Métricas en el conjunto de prueba (20% reservado, 960 reseñas)

| Métrica | Valor |
|---|---|
| F1-macro | **0.6688** |
| Accuracy | 0.676 |
| Balanced accuracy | 0.6756 |
| F1 ponderado | 0.6719 |

Config: `lr=3e-5`, entrenado sobre 2685 reseñas etiquetadas de forma semisupervisada.

## Archivos

- `model.safetensors` — pesos del modelo (Git LFS)
- `config.json` — arquitectura + mapeo de clases (`id2label`)
- `tokenizer.json`, `tokenizer_config.json` — tokenizador

## Uso

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ruta = "modelo_entrenado/beto_sentimiento"
tok = AutoTokenizer.from_pretrained(ruta)
modelo = AutoModelForSequenceClassification.from_pretrained(ruta)

# Clases: muy negativo / negativo / neutral / positivo / muy positivo
import torch
inputs = tok("El servicio fue excelente, muy recomendado", return_tensors="pt", truncation=True)
pred = modelo(**inputs).logits.argmax(-1).item()
print(modelo.config.id2label[pred])
```

También disponible como GitHub Release: `modelo-beto-v2`.
Regenerable con `scripts/15_entrenamiento_final/entrenar_final.py`.
