"""Entrenadores compartidos para el flujo semisupervisado (fases 13, 14 y 15).

La validacion cruzada (fase 13), el self-training (fase 14) y el entrenamiento final
(fase 15) entrenan los MISMOS seis modelos del proyecto con hiperparametros variables:

    clasicos      -> SVM (LinearSVC), Naive Bayes (MultinomialNB)  sobre TF-IDF
    deep learning -> TextCNN, BiLSTM                               (PyTorch)
    transformers  -> BETO, XLM-RoBERTa                             (HuggingFace)

Este modulo centraliza el entrenamiento y la prediccion para que las tres fases usen
exactamente el mismo codigo. Reutiliza las arquitecturas y utilidades ya definidas en
las fases 07 y 08 (cargadas por ruta, porque los paquetes con prefijo numerico no se
pueden importar con `import` normal).

Todas las funciones de entrenamiento de redes/transformers devuelven un `historial`
por epoca (perdida de entrenamiento y F1-macro de validacion) para graficar las
curvas de entrenamiento pedidas en la retroalimentacion del docente.

Las estrategias de desbalance quedan fijas en lo que gano en el flujo anterior:
class_weight para redes y transformers; para clasicos, class_weight balanced (SVM)
y SMOTE opcional (Naive Bayes) como hiperparametro.
"""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "scripts"))
from _comun.evaluacion import CLASES  # noqa: E402


def _cargar_modulo(ruta, nombre):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


_fase07 = _cargar_modulo(PROJECT_ROOT / "scripts" / "07_modelado" / "entrenar_evaluar.py", "fase07")
_fase08 = _cargar_modulo(PROJECT_ROOT / "scripts" / "08_dl" / "entrenar_dl.py", "fase08")

construir_vectorizador = _fase07.construir_vectorizador
aplicar_smote = _fase07.aplicar_smote
construir_vocabulario = _fase08.construir_vocabulario
codificar = _fase08.codificar
pesos_de_clase = _fase08.pesos_de_clase
f1_macro_rapido = _fase08.f1_macro_rapido
TextCNN = _fase08.TextCNN
BiLSTM = _fase08.BiLSTM

# Nombre legible -> id de HuggingFace Hub (mismos de la fase 09).
MODELOS_HF = {
    "beto": "dccuchile/bert-base-spanish-wwm-cased",
    "xlm_roberta": "xlm-roberta-base",
}

FAMILIAS = {
    "svm": "clasico", "naive_bayes": "clasico",
    "cnn": "deep_learning", "lstm": "deep_learning",
    "beto": "transformer", "xlm_roberta": "transformer",
}

# Columna de texto por familia (misma convencion que el flujo anterior).
COLUMNA_TEXTO = {
    "clasico": "texto_modelo",
    "deep_learning": "texto_modelo",
    "transformer": "comentario_limpio",
}

CLASE_A_IDX = {c: i for i, c in enumerate(CLASES)}


def dispositivo(forzar_cpu=False):
    return torch.device("cpu" if forzar_cpu or not torch.cuda.is_available() else "cuda")


# --------------------------------------------------------------------------- #
# Clasicos (SVM y Naive Bayes sobre TF-IDF)
# --------------------------------------------------------------------------- #
def entrenar_clasico(algoritmo, x_train_txt, y_train, C=1.0, alpha=1.0,
                     max_features=20000, usar_smote=False, random_state=42):
    """Entrena un clasico y devuelve {vectorizador, clasificador}."""
    vectorizador = construir_vectorizador(max_features, usar_stopwords=True)
    x_tr = vectorizador.fit_transform(x_train_txt)
    y_fit = y_train
    if usar_smote:
        x_tr, y_fit = aplicar_smote(x_tr, y_train, random_state)

    if algoritmo == "svm":
        clasificador = LinearSVC(C=C, class_weight="balanced", max_iter=2000)
    elif algoritmo == "naive_bayes":
        clasificador = MultinomialNB(alpha=alpha)
    else:
        raise ValueError(f"Clasico desconocido: {algoritmo}")
    clasificador.fit(x_tr, y_fit)
    return {"vectorizador": vectorizador, "clasificador": clasificador}


def predecir_clasico(artefacto, textos):
    """Devuelve (predicciones_texto, scores, orden_clases_scores)."""
    x = artefacto["vectorizador"].transform(textos)
    clf = artefacto["clasificador"]
    pred = list(clf.predict(x))
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(x)
    else:
        margenes = clf.decision_function(x)
        margenes = margenes - margenes.max(axis=1, keepdims=True)
        exp = np.exp(margenes)
        scores = exp / exp.sum(axis=1, keepdims=True)  # softmax sobre margenes
    return pred, scores, list(clf.classes_)


# --------------------------------------------------------------------------- #
# Redes (TextCNN y BiLSTM) con perdida ponderada por clase
# --------------------------------------------------------------------------- #
def entrenar_red(arquitectura, x_train_txt, y_train_idx, x_valid_txt, y_valid_idx,
                 device, lr=1e-3, embed_dim=100, num_filtros=100, hidden=128,
                 dropout=0.5, epochs=30, paciencia=5, batch_size=32, max_len=80,
                 min_freq=2, random_state=42, verbose=False):
    """Entrena CNN/LSTM con early stopping. Devuelve (modelo, vocab, historial)."""
    torch.manual_seed(random_state)
    vocab = construir_vocabulario(x_train_txt, min_freq)
    x_tr = codificar(list(x_train_txt), vocab, max_len)
    x_va = codificar(list(x_valid_txt), vocab, max_len)
    num_classes = len(CLASES)

    if arquitectura == "cnn":
        modelo = TextCNN(len(vocab), embed_dim, num_classes,
                         num_filtros=num_filtros, dropout=dropout).to(device)
    elif arquitectura == "lstm":
        modelo = BiLSTM(len(vocab), embed_dim, num_classes,
                        hidden=hidden, dropout=dropout).to(device)
    else:
        raise ValueError(f"Red desconocida: {arquitectura}")

    peso = pesos_de_clase(np.asarray(y_train_idx), num_classes, device)
    criterio = nn.CrossEntropyLoss(weight=peso)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(np.asarray(y_train_idx))),
        batch_size=batch_size, shuffle=True,
    )
    valid_loader = DataLoader(TensorDataset(torch.from_numpy(x_va)), batch_size=batch_size)

    historial, mejor_f1, mejor_estado, sin_mejora = [], -1.0, None, 0
    for epoca in range(1, epochs + 1):
        modelo.train()
        perdidas = []
        for xb, yb in train_loader:
            optimizador.zero_grad()
            loss = criterio(modelo(xb.to(device)), yb.to(device))
            loss.backward()
            optimizador.step()
            perdidas.append(float(loss.item()))

        pred_valid = _predecir_idx_red(modelo, valid_loader, device)
        f1_valid = f1_macro_rapido(np.asarray(y_valid_idx), pred_valid, num_classes)
        historial.append({"epoca": epoca, "loss_train": round(float(np.mean(perdidas)), 4),
                          "f1_valid": round(f1_valid, 4)})
        if verbose:
            print(f"   epoca {epoca:>2} | loss {historial[-1]['loss_train']:.4f} "
                  f"| valid F1-macro {f1_valid:.4f}")
        if f1_valid > mejor_f1:
            mejor_f1, mejor_estado, sin_mejora = f1_valid, deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
        if sin_mejora >= paciencia:
            break

    modelo.load_state_dict(mejor_estado)
    return modelo, vocab, historial


def _predecir_idx_red(modelo, loader, device):
    modelo.eval()
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(modelo(xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def predecir_red(modelo, vocab, textos, device, max_len=80, batch_size=64):
    """Devuelve (pred_idx, scores softmax) para una lista de textos."""
    x = codificar(list(textos), vocab, max_len)
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size)
    modelo.eval()
    scores = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = modelo(xb.to(device))
            scores.append(torch.softmax(logits, dim=1).cpu().numpy())
    scores = np.concatenate(scores)
    return scores.argmax(axis=1), scores


# --------------------------------------------------------------------------- #
# Transformers (BETO y XLM-RoBERTa) con AMP y perdida ponderada
# --------------------------------------------------------------------------- #
def entrenar_transformer(nombre, x_train_txt, y_train_idx, x_valid_txt, y_valid_idx,
                         device, lr=2e-5, epochs=4, paciencia=2, batch_size=16,
                         max_len=128, random_state=42, verbose=False):
    """Fine-tuning con early stopping. Devuelve (modelo, tokenizer, historial)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    try:
        from transformers import get_linear_schedule_with_warmup
    except ImportError:
        get_linear_schedule_with_warmup = None

    torch.manual_seed(random_state)
    hf_id = MODELOS_HF[nombre]
    num_classes = len(CLASES)
    use_amp = device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    modelo = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=num_classes,
        id2label={i: c for i, c in enumerate(CLASES)},
        label2id={c: i for i, c in enumerate(CLASES)},
    ).to(device)

    def tokenizar(textos):
        enc = tokenizer(list(textos), truncation=True, padding="max_length",
                        max_length=max_len, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    train_ids, train_attn = tokenizar(x_train_txt)
    valid_ids, valid_attn = tokenizar(x_valid_txt)

    peso = pesos_de_clase(np.asarray(y_train_idx), num_classes, device)
    criterio = nn.CrossEntropyLoss(weight=peso)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)

    train_loader = DataLoader(
        TensorDataset(train_ids, train_attn, torch.from_numpy(np.asarray(y_train_idx))),
        batch_size=batch_size, shuffle=True,
    )
    valid_loader = DataLoader(TensorDataset(valid_ids, valid_attn), batch_size=batch_size)

    total_pasos = len(train_loader) * epochs
    scheduler = None
    if get_linear_schedule_with_warmup is not None:
        scheduler = get_linear_schedule_with_warmup(
            optimizador, num_warmup_steps=int(0.1 * total_pasos), num_training_steps=total_pasos,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    historial, mejor_f1, mejor_estado, sin_mejora = [], -1.0, None, 0
    for epoca in range(1, epochs + 1):
        modelo.train()
        perdidas = []
        for input_ids, attn, yb in train_loader:
            optimizador.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = modelo(input_ids=input_ids.to(device),
                                attention_mask=attn.to(device)).logits
                loss = criterio(logits, yb.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizador)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            scaler.step(optimizador)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            perdidas.append(float(loss.item()))

        pred_valid, _ = predecir_transformer(modelo, tokenizer, x_valid_txt, device,
                                             max_len=max_len, batch_size=batch_size)
        f1_valid = f1_macro_rapido(np.asarray(y_valid_idx), pred_valid, num_classes)
        historial.append({"epoca": epoca, "loss_train": round(float(np.mean(perdidas)), 4),
                          "f1_valid": round(f1_valid, 4)})
        if verbose:
            print(f"   epoca {epoca} | loss {historial[-1]['loss_train']:.4f} "
                  f"| valid F1-macro {f1_valid:.4f}")
        if f1_valid > mejor_f1:
            mejor_f1, mejor_estado, sin_mejora = f1_valid, deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
        if sin_mejora >= paciencia:
            break

    modelo.load_state_dict(mejor_estado)
    return modelo, tokenizer, historial


def predecir_transformer(modelo, tokenizer, textos, device, max_len=128, batch_size=32):
    """Devuelve (pred_idx, scores softmax) para una lista de textos."""
    enc = tokenizer(list(textos), truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")
    loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"]),
                        batch_size=batch_size)
    use_amp = device.type == "cuda"
    modelo.eval()
    scores = []
    with torch.no_grad():
        for input_ids, attn in loader:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = modelo(input_ids=input_ids.to(device),
                                attention_mask=attn.to(device)).logits
            scores.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    scores = np.concatenate(scores)
    return scores.argmax(axis=1), scores


# --------------------------------------------------------------------------- #
# Interfaz unificada por nombre de modelo
# --------------------------------------------------------------------------- #
def entrenar_modelo(nombre, config, x_train_txt, y_train_txt, x_valid_txt, y_valid_txt,
                    device, random_state=42, verbose=False):
    """Entrena cualquiera de los 6 modelos y devuelve un artefacto homogeneo.

    `config` es el dict de hiperparametros del modelo (los del grid de la fase 13).
    Devuelve {"nombre", "familia", "config", "historial", ...artefactos propios...}.
    """
    familia = FAMILIAS[nombre]
    if familia == "clasico":
        artefacto = entrenar_clasico(nombre, x_train_txt, list(y_train_txt),
                                     random_state=random_state, **config)
        return {"nombre": nombre, "familia": familia, "config": config,
                "historial": [], **artefacto}

    y_train_idx = np.array([CLASE_A_IDX[c] for c in y_train_txt])
    y_valid_idx = np.array([CLASE_A_IDX[c] for c in y_valid_txt])
    if familia == "deep_learning":
        modelo, vocab, historial = entrenar_red(
            nombre, x_train_txt, y_train_idx, x_valid_txt, y_valid_idx, device,
            random_state=random_state, verbose=verbose, **config)
        return {"nombre": nombre, "familia": familia, "config": config,
                "historial": historial, "modelo": modelo, "vocab": vocab}

    modelo, tokenizer, historial = entrenar_transformer(
        nombre, x_train_txt, y_train_idx, x_valid_txt, y_valid_idx, device,
        random_state=random_state, verbose=verbose, **config)
    return {"nombre": nombre, "familia": familia, "config": config,
            "historial": historial, "modelo": modelo, "tokenizer": tokenizer}


def predecir_modelo(artefacto, textos, device):
    """Prediccion homogenea: devuelve (predicciones_texto, scores, orden_clases)."""
    familia = artefacto["familia"]
    if familia == "clasico":
        return predecir_clasico(artefacto, textos)
    if familia == "deep_learning":
        pred_idx, scores = predecir_red(artefacto["modelo"], artefacto["vocab"],
                                        textos, device)
    else:
        pred_idx, scores = predecir_transformer(artefacto["modelo"], artefacto["tokenizer"],
                                                textos, device)
    return [CLASES[i] for i in pred_idx], scores, list(CLASES)
