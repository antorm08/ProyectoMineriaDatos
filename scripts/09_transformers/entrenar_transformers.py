"""Fase 09: modelos Transformer (BETO y XLM-RoBERTa) con fine-tuning.

Esta fase cubre la familia de transformers del proyecto:

    BETO         -> dccuchile/bert-base-spanish-wwm-cased
                    BERT preentrenado en espanol (Universidad de Chile).
    XLM-RoBERTa  -> xlm-roberta-base
                    RoBERTa multilingue preentrenado en 100 idiomas.

Se hace fine-tuning de cada modelo sobre las resenas peruanas. A diferencia de los
clasicos (TF-IDF) y las redes (embeddings desde cero), los transformers usan su propio
tokenizador subpalabra y conocimiento preentrenado, por lo que se alimentan con el
**texto natural** (`comentario_limpio`, con mayusculas y signos) en vez del texto
minusculizado. El conjunto de filas evaluadas es el mismo que en las demas familias
(ver scripts/_comun/datos.py), de modo que la comparacion es justa.

Estrategia frente al desbalance: CrossEntropyLoss ponderada por frecuencia inversa de
clase (class_weight). Es el equivalente directo a la estrategia "balanced" de los
clasicos. Se puede agregar la variante "base" con --estrategias base class_weight.

Aceleracion: usa GPU (CUDA) si esta disponible, con precision mixta automatica (AMP)
para entrar en 8 GB de VRAM y entrenar mas rapido. Cae a CPU si no hay GPU.

Entrada:
    data/splits/train.csv, valid.csv, test.csv  (columnas comentario_limpio y sentimiento_final)

Salidas (reports/08_transformers/):
    comparacion_transformers.csv
    f1_por_clase_transformers.csv
    matriz_confusion_<modelo>_<estrategia>_test.csv / .png
    reporte_clasificacion_<modelo>_<estrategia>_test.txt
models/:
    mejor_modelo_transformer/   (modelo + tokenizador de la mejor combinacion)
"""

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:  # nombre o ubicacion distinta segun version
    get_linear_schedule_with_warmup = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
from _comun.datos import cargar_split  # noqa: E402
from _comun.evaluacion import CLASES, evaluar_split  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "08_transformers"
MODELS_DIR = PROJECT_ROOT / "models"
MODELO_DIR = MODELS_DIR / "mejor_modelo_transformer"

FAMILIA = "transformer"
COLUMNA_TEXTO = "comentario_limpio"

# Nombre legible -> identificador en HuggingFace Hub.
MODELOS = {
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "xlm_roberta": "xlm-roberta-base",
}
ESTRATEGIAS = ["base", "class_weight"]


def pesos_de_clase(y_idx, num_classes, device):
    """Peso por clase = N / (num_classes * conteo_clase). Frecuencia inversa."""
    conteos = np.bincount(y_idx, minlength=num_classes)
    pesos = len(y_idx) / (num_classes * np.maximum(conteos, 1))
    return torch.tensor(pesos, dtype=torch.float32, device=device)


def tokenizar(tokenizer, textos, max_len):
    enc = tokenizer(
        list(textos),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


def predecir_idx(modelo, loader, device, use_amp):
    modelo.eval()
    predicciones = []
    with torch.no_grad():
        for input_ids, attn in loader:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = modelo(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
            predicciones.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predicciones)


def f1_macro_rapido(y_true_idx, y_pred_idx, num_classes):
    """F1-macro ligero (sin sklearn) para el early stopping por epoca."""
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred_idx == c) & (y_true_idx == c))
        fp = np.sum((y_pred_idx == c) & (y_true_idx != c))
        fn = np.sum((y_pred_idx != c) & (y_true_idx == c))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


def entrenar_transformer(nombre, hf_id, estrategia, tensores, y_idx, device, args):
    """Fine-tuning de un transformer. Devuelve (pred_valid, pred_test, modelo, tokenizer)."""
    (train_ids, train_attn, valid_ids, valid_attn, test_ids, test_attn) = tensores
    y_train_idx, y_valid_idx = y_idx
    num_classes = len(CLASES)
    use_amp = device.type == "cuda"

    id2label = {i: c for i, c in enumerate(CLASES)}
    label2id = {c: i for i, c in enumerate(CLASES)}

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=num_classes, id2label=id2label, label2id=label2id,
    ).to(device)

    peso = pesos_de_clase(y_train_idx, num_classes, device) if estrategia == "class_weight" else None
    criterio = nn.CrossEntropyLoss(weight=peso)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.01)

    train_loader = DataLoader(
        TensorDataset(train_ids, train_attn, torch.from_numpy(y_train_idx)),
        batch_size=args.batch_size, shuffle=True,
    )
    valid_loader = DataLoader(TensorDataset(valid_ids, valid_attn), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(test_ids, test_attn), batch_size=args.batch_size)

    total_pasos = len(train_loader) * args.epochs
    scheduler = None
    if get_linear_schedule_with_warmup is not None:
        scheduler = get_linear_schedule_with_warmup(
            optimizador, num_warmup_steps=int(0.1 * total_pasos), num_training_steps=total_pasos,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    mejor_f1, mejor_estado, sin_mejora = -1.0, None, 0
    for epoca in range(1, args.epochs + 1):
        modelo.train()
        for input_ids, attn, yb in train_loader:
            optimizador.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = modelo(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
                loss = criterio(logits, yb.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizador)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            scaler.step(optimizador)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

        f1_valid = f1_macro_rapido(y_valid_idx, predecir_idx(modelo, valid_loader, device, use_amp), num_classes)
        if f1_valid > mejor_f1:
            mejor_f1, mejor_estado, sin_mejora = f1_valid, deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
        print(f"   epoca {epoca}/{args.epochs} | valid F1-macro {f1_valid:.4f}"
              + ("  * (mejor)" if sin_mejora == 0 else ""))
        if sin_mejora >= args.paciencia:
            print(f"   early stopping en epoca {epoca} (sin mejora en {args.paciencia} epocas)")
            break

    modelo.load_state_dict(mejor_estado)
    pred_valid = predecir_idx(modelo, valid_loader, device, use_amp)
    pred_test = predecir_idx(modelo, test_loader, device, use_amp)
    return pred_valid, pred_test, modelo, tokenizer


def entrenar_y_evaluar(args):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cpu":
        print("AVISO: sin GPU el fine-tuning de transformers es lento (puede tardar bastante).")

    x_train_txt, y_train = cargar_split("train", columna_texto=COLUMNA_TEXTO)
    x_valid_txt, y_valid = cargar_split("valid", columna_texto=COLUMNA_TEXTO)
    x_test_txt, y_test = cargar_split("test", columna_texto=COLUMNA_TEXTO)
    print(f"Datos: {len(x_train_txt)} train | {len(x_valid_txt)} valid | {len(x_test_txt)} test "
          f"| columna='{COLUMNA_TEXTO}' | max_len={args.max_len}")

    clase_a_idx = {clase: i for i, clase in enumerate(CLASES)}
    y_train_idx = y_train.map(clase_a_idx).to_numpy()
    y_valid_idx = y_valid.map(clase_a_idx).to_numpy()
    y_valid_txt, y_test_txt = y_valid.tolist(), y_test.tolist()

    filas_comparacion, filas_f1_clase, resultados = [], [], {}
    for nombre in args.modelos:
        hf_id = MODELOS[nombre]
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        # La tokenizacion depende del modelo, por eso se hace dentro del bucle.
        train_ids, train_attn = tokenizar(tokenizer, x_train_txt, args.max_len)
        valid_ids, valid_attn = tokenizar(tokenizer, x_valid_txt, args.max_len)
        test_ids, test_attn = tokenizar(tokenizer, x_test_txt, args.max_len)
        tensores = (train_ids, train_attn, valid_ids, valid_attn, test_ids, test_attn)

        for estrategia in args.estrategias:
            print(f"\n>> Fine-tuning {nombre} ({hf_id}) | estrategia: {estrategia}")
            pred_valid_idx, pred_test_idx, modelo, tok = entrenar_transformer(
                nombre, hf_id, estrategia, tensores, (y_train_idx, y_valid_idx), device, args,
            )
            pred_valid = [CLASES[i] for i in pred_valid_idx]
            pred_test = [CLASES[i] for i in pred_test_idx]

            fila_valid = evaluar_split(y_valid_txt, pred_valid, FAMILIA, nombre, estrategia,
                                       "valid", REPORT_DIR, filas_comparacion, filas_f1_clase)
            fila_test = evaluar_split(y_test_txt, pred_test, FAMILIA, nombre, estrategia,
                                      "test", REPORT_DIR, filas_comparacion, filas_f1_clase)
            resultados[(nombre, estrategia)] = {
                "modelo": modelo, "tokenizer": tok, "valid": fila_valid, "test": fila_test,
            }
            print(f"   -> valid F1-macro {fila_valid['f1_macro']:.4f} | test F1-macro {fila_test['f1_macro']:.4f}")

            # Liberar VRAM antes del siguiente modelo.
            if device.type == "cuda":
                torch.cuda.empty_cache()

    comparacion = pd.DataFrame(filas_comparacion)
    comparacion.to_csv(REPORT_DIR / "comparacion_transformers.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_f1_clase).to_csv(REPORT_DIR / "f1_por_clase_transformers.csv", index=False, encoding="utf-8-sig")

    # La mejor combinacion se elige por F1-macro en validacion y se guarda completa.
    mejor = max(resultados, key=lambda combo: resultados[combo]["valid"]["f1_macro"])
    MODELO_DIR.mkdir(parents=True, exist_ok=True)
    resultados[mejor]["modelo"].save_pretrained(MODELO_DIR)
    resultados[mejor]["tokenizer"].save_pretrained(MODELO_DIR)

    print("\n" + "=" * 78)
    print("COMPARACION TRANSFORMERS (ordenada por F1-macro en test)")
    print("=" * 78)
    tabla_test = comparacion[comparacion["split"] == "test"].sort_values("f1_macro", ascending=False)
    print(tabla_test.to_string(index=False))
    print(f"\nMejor transformer por F1-macro en validacion: {mejor[0]} + {mejor[1]} "
          f"(valid {resultados[mejor]['valid']['f1_macro']:.4f} | test {resultados[mejor]['test']['f1_macro']:.4f})")
    print(f"Modelo guardado en: {MODELO_DIR}")
    print(f"Reportes en: {REPORT_DIR}")


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Fine-tuning y evaluacion de transformers (BETO y XLM-RoBERTa, fase 09).")
    parser.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=list(MODELOS))
    parser.add_argument("--estrategias", nargs="+", default=["class_weight"], choices=ESTRATEGIAS,
                        help="Por defecto solo class_weight (mas rapido). Usa 'base class_weight' para comparar.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--paciencia", type=int, default=2, help="Early stopping: epocas sin mejora.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128, help="Maximo de tokens por resena.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Forzar CPU aunque haya GPU.")
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)
    entrenar_y_evaluar(args)


if __name__ == "__main__":
    main()
