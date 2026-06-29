"""Fase 08: modelos de Deep Learning (CNN y LSTM).

Esta fase cubre la familia de aprendizaje profundo del proyecto:

    CNN  -> TextCNN (Kim, 2014): embeddings + convoluciones 1D con varios tamanos de
            kernel (3, 4, 5) + max-pooling temporal + capa densa.
    LSTM -> BiLSTM: embeddings + LSTM bidireccional + mean-pooling enmascarado + capa densa.

Ambos parten de los mismos splits que las demas familias (fase 06) y se evaluan con el
modulo comun scripts/_comun/evaluacion.py, por lo que las metricas son comparables con
los clasicos (fase 07) y los transformers (fase 09).

A diferencia de TF-IDF, aqui el texto se representa como una secuencia de indices de
palabras y la red aprende sus propios embeddings desde cero (no hay vectores
preentrenados, coherente con un dataset pequeno y especifico).

Estrategias frente al desbalance:
    base         -> perdida sin ponderar
    class_weight -> CrossEntropyLoss ponderada por frecuencia inversa de clase

(No se usa SMOTE: sobre secuencias de longitud variable no aplica de forma natural; el
equivalente estandar en deep learning es ponderar la perdida, que es lo que se hace.)

Entrada:
    data/splits/train.csv, valid.csv, test.csv  (columnas texto_modelo y sentimiento_final)

Salidas (reports/07_dl/):
    comparacion_dl.csv
    f1_por_clase_dl.csv
    matriz_confusion_<modelo>_<estrategia>_test.csv / .png
    reporte_clasificacion_<modelo>_<estrategia>_test.txt
models/:
    mejor_modelo_dl.pt   (mejor combinacion DL: pesos + vocabulario + config)
"""

import argparse
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
from _comun.datos import cargar_split  # noqa: E402
from _comun.evaluacion import CLASES, evaluar_split  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports" / "07_dl"
MODELS_DIR = PROJECT_ROOT / "models"
MODELO_FILE = MODELS_DIR / "mejor_modelo_dl.pt"

FAMILIA = "deep_learning"
MODELOS = ["cnn", "lstm"]
ESTRATEGIAS = ["base", "class_weight"]

PAD, UNK = "<pad>", "<unk>"
IDX_PAD = 0


# --------------------------------------------------------------------------- #
# Vocabulario y codificacion de texto
# --------------------------------------------------------------------------- #
def construir_vocabulario(textos, min_freq):
    """Construye word2idx desde los textos de entrenamiento (tokeniza por espacios)."""
    contador = Counter()
    for texto in textos:
        contador.update(texto.split())
    # Indices reservados: 0 -> <pad>, 1 -> <unk>.
    vocab = {PAD: IDX_PAD, UNK: 1}
    for palabra, freq in contador.most_common():
        if freq >= min_freq:
            vocab[palabra] = len(vocab)
    return vocab


def codificar(textos, vocab, max_len):
    """Convierte textos a una matriz (n, max_len) de indices, con padding/truncado."""
    idx_unk = vocab[UNK]
    matriz = np.full((len(textos), max_len), IDX_PAD, dtype=np.int64)
    for i, texto in enumerate(textos):
        ids = [vocab.get(palabra, idx_unk) for palabra in texto.split()][:max_len]
        matriz[i, : len(ids)] = ids
    return matriz


# --------------------------------------------------------------------------- #
# Arquitecturas
# --------------------------------------------------------------------------- #
class TextCNN(nn.Module):
    """CNN para texto con varios tamanos de kernel y max-pooling temporal."""

    def __init__(self, vocab_size, embed_dim, num_classes, kernels=(3, 4, 5), num_filtros=100, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=IDX_PAD)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filtros, kernel_size=k) for k in kernels]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filtros * len(kernels), num_classes)

    def forward(self, x):
        emb = self.embedding(x).transpose(1, 2)          # (B, embed_dim, T)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))                    # (B, num_filtros, T-k+1)
            pooled.append(torch.max(c, dim=2).values)    # (B, num_filtros)
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


class BiLSTM(nn.Module):
    """LSTM bidireccional con mean-pooling enmascarado sobre los tokens reales."""

    def __init__(self, vocab_size, embed_dim, num_classes, hidden=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=IDX_PAD)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        mask = (x != IDX_PAD).unsqueeze(-1).float()      # (B, T, 1)
        emb = self.embedding(x)
        salidas, _ = self.lstm(emb)                      # (B, T, 2H)
        # Mean-pooling solo sobre tokens reales (evita que el padding diluya la senal).
        sumas = (salidas * mask).sum(dim=1)
        longitudes = mask.sum(dim=1).clamp(min=1.0)
        promedio = sumas / longitudes
        return self.fc(self.dropout(promedio))


def construir_modelo(nombre, vocab_size, embed_dim, num_classes):
    if nombre == "cnn":
        return TextCNN(vocab_size, embed_dim, num_classes)
    if nombre == "lstm":
        return BiLSTM(vocab_size, embed_dim, num_classes)
    raise ValueError(f"Modelo DL desconocido: {nombre}")


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def pesos_de_clase(y_idx, num_classes, device):
    """Peso por clase = N / (num_classes * conteo_clase). Frecuencia inversa."""
    conteos = np.bincount(y_idx, minlength=num_classes)
    pesos = len(y_idx) / (num_classes * np.maximum(conteos, 1))
    return torch.tensor(pesos, dtype=torch.float32, device=device)


def predecir_idx(modelo, loader, device):
    modelo.eval()
    predicciones = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = modelo(xb.to(device))
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


def entrenar_modelo(nombre, estrategia, datos, device, args):
    """Entrena una combinacion modelo+estrategia y devuelve (pred_valid, pred_test, modelo)."""
    x_train, y_train_idx, x_valid, y_valid_idx, x_test = datos
    num_classes = len(CLASES)

    torch.manual_seed(args.random_state)
    modelo = construir_modelo(nombre, args.vocab_size, args.embed_dim, num_classes).to(device)

    peso = pesos_de_clase(y_train_idx, num_classes, device) if estrategia == "class_weight" else None
    criterio = nn.CrossEntropyLoss(weight=peso)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=args.lr)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_idx)),
        batch_size=args.batch_size, shuffle=True,
    )
    valid_loader = DataLoader(TensorDataset(torch.from_numpy(x_valid)), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test)), batch_size=args.batch_size)

    mejor_f1 = -1.0
    mejor_estado = None
    sin_mejora = 0
    for epoca in range(1, args.epochs + 1):
        modelo.train()
        for xb, yb in train_loader:
            optimizador.zero_grad()
            loss = criterio(modelo(xb.to(device)), yb.to(device))
            loss.backward()
            optimizador.step()

        f1_valid = f1_macro_rapido(y_valid_idx, predecir_idx(modelo, valid_loader, device), num_classes)
        if f1_valid > mejor_f1:
            mejor_f1, mejor_estado, sin_mejora = f1_valid, deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
        print(f"   epoca {epoca:>2} | valid F1-macro {f1_valid:.4f}"
              + (f" *  (mejor)" if sin_mejora == 0 else ""))
        if sin_mejora >= args.paciencia:
            print(f"   early stopping en epoca {epoca} (sin mejora en {args.paciencia} epocas)")
            break

    modelo.load_state_dict(mejor_estado)
    return predecir_idx(modelo, valid_loader, device), predecir_idx(modelo, test_loader, device), modelo


def entrenar_y_evaluar(args):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    x_train_txt, y_train = cargar_split("train")
    x_valid_txt, y_valid = cargar_split("valid")
    x_test_txt, y_test = cargar_split("test")

    vocab = construir_vocabulario(x_train_txt, args.min_freq)
    args.vocab_size = len(vocab)
    print(f"Vocabulario: {args.vocab_size} palabras (min_freq={args.min_freq}) | max_len={args.max_len}")

    x_train = codificar(x_train_txt, vocab, args.max_len)
    x_valid = codificar(x_valid_txt, vocab, args.max_len)
    x_test = codificar(x_test_txt, vocab, args.max_len)

    clase_a_idx = {clase: i for i, clase in enumerate(CLASES)}
    y_train_idx = y_train.map(clase_a_idx).to_numpy()
    y_valid_idx = y_valid.map(clase_a_idx).to_numpy()
    y_test_idx = y_test.map(clase_a_idx).to_numpy()
    datos = (x_train, y_train_idx, x_valid, y_valid_idx, x_test)

    y_valid_txt = y_valid.tolist()
    y_test_txt = y_test.tolist()

    filas_comparacion, filas_f1_clase, modelos = [], [], {}
    for nombre in args.modelos:
        for estrategia in args.estrategias:
            print(f"\n>> Entrenando {nombre} ({estrategia})")
            pred_valid_idx, pred_test_idx, modelo = entrenar_modelo(nombre, estrategia, datos, device, args)
            pred_valid = [CLASES[i] for i in pred_valid_idx]
            pred_test = [CLASES[i] for i in pred_test_idx]

            fila_valid = evaluar_split(y_valid_txt, pred_valid, FAMILIA, nombre, estrategia,
                                       "valid", REPORT_DIR, filas_comparacion, filas_f1_clase)
            fila_test = evaluar_split(y_test_txt, pred_test, FAMILIA, nombre, estrategia,
                                      "test", REPORT_DIR, filas_comparacion, filas_f1_clase)
            modelos[(nombre, estrategia)] = {"modelo": modelo, "valid": fila_valid, "test": fila_test}
            print(f"   -> valid F1-macro {fila_valid['f1_macro']:.4f} | test F1-macro {fila_test['f1_macro']:.4f}")

    comparacion = pd.DataFrame(filas_comparacion)
    comparacion.to_csv(REPORT_DIR / "comparacion_dl.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(filas_f1_clase).to_csv(REPORT_DIR / "f1_por_clase_dl.csv", index=False, encoding="utf-8-sig")

    # La mejor combinacion DL se elige por F1-macro en validacion.
    mejor = max(modelos, key=lambda combo: modelos[combo]["valid"]["f1_macro"])
    torch.save({
        "state_dict": modelos[mejor]["modelo"].state_dict(),
        "arquitectura": mejor[0],
        "estrategia": mejor[1],
        "vocab": vocab,
        "clases": CLASES,
        "config": {"embed_dim": args.embed_dim, "max_len": args.max_len},
    }, MODELO_FILE)

    print("\n" + "=" * 78)
    print("COMPARACION DEEP LEARNING (ordenada por F1-macro en test)")
    print("=" * 78)
    tabla_test = comparacion[comparacion["split"] == "test"].sort_values("f1_macro", ascending=False)
    print(tabla_test.to_string(index=False))
    print(f"\nMejor combinacion DL por F1-macro en validacion: {mejor[0]} + {mejor[1]} "
          f"(valid {modelos[mejor]['valid']['f1_macro']:.4f} | test {modelos[mejor]['test']['f1_macro']:.4f})")
    print(f"Modelo guardado en: {MODELO_FILE}")
    print(f"Reportes en: {REPORT_DIR}")


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Entrena y evalua modelos de deep learning (CNN y LSTM, fase 08).")
    parser.add_argument("--modelos", nargs="+", default=MODELOS, choices=MODELOS)
    parser.add_argument("--estrategias", nargs="+", default=ESTRATEGIAS, choices=ESTRATEGIAS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--paciencia", type=int, default=5, help="Early stopping: epocas sin mejora.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=80, help="Maximo de tokens por resena.")
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--min-freq", type=int, default=2, help="Frecuencia minima para entrar al vocabulario.")
    parser.add_argument("--lr", type=float, default=1e-3)
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
