"""Cliente minimo para LLMs externos (NVIDIA NIM y OpenRouter) usado en el etiquetado.

El flujo semisupervisado (fases 11-14) usa un LLM como anotador asistido: etiqueta la
semilla de 500 registros y el conjunto de prueba, y arbitra los casos dudosos del
self-training. Este modulo centraliza:

    - carga de claves desde .env (NUNCA versionado; ver .gitignore)
    - llamadas chat/completions con reintentos y backoff (429/5xx/timeouts)
    - el prompt de etiquetado con la guia de las 5 clases ordinales
    - etiquetado por lotes con salida JSON validada

Proveedores:
    nvidia     -> https://integrate.api.nvidia.com/v1  (gratuito con rate limit; principal)
    openrouter -> https://openrouter.ai/api/v1         (free tier; fallback)
"""

import json
import os
import re
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]

URLS = {
    "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}
ENV_KEYS = {
    "nvidia": "NVIDIA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

CLASES = ["muy negativo", "negativo", "neutral", "positivo", "muy positivo"]

PROMPT_SISTEMA = (
    "Eres un anotador experto de sentimiento en resenas de consumidores peruanos de Google Maps. "
    "Etiquetas cada resena con UNA de cinco clases ordinales: "
    "muy negativo, negativo, neutral, positivo, muy positivo."
)

GUIA_CLASES = """Criterios de etiquetado:
- "muy negativo": enojo o queja grave (estafa, maltrato, perdida de dinero/tiempo), recomienda no ir.
- "negativo": insatisfaccion clara pero moderada; predomina lo malo sobre lo bueno.
- "neutral": mixto o informativo; elogios y quejas equilibrados; sin polaridad dominante.
- "positivo": satisfaccion moderada; elogio con peros o sin entusiasmo.
- "muy positivo": entusiasmo claro; recomienda con fuerza; elogio sin reservas.
Juzga SOLO por el texto de la resena (no dispones de las estrellas).
Responde UNICAMENTE con un array JSON valido, un objeto por resena y en el mismo orden:
[{"id": 0, "etiqueta": "<clase>", "confianza": 0.0, "justificacion": "maximo 15 palabras"}]"""


def _cargar_env():
    """Carga variables desde .env (raiz del proyecto) sin pisar las ya definidas."""
    ruta = PROJECT_ROOT / ".env"
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        os.environ.setdefault(clave.strip(), valor.strip())


def api_key(proveedor):
    _cargar_env()
    clave = os.environ.get(ENV_KEYS[proveedor], "")
    if not clave:
        raise RuntimeError(f"Falta {ENV_KEYS[proveedor]} en el entorno o en .env")
    return clave


def completar(mensajes, modelo, proveedor="nvidia", max_tokens=4096, temperature=0.2,
              intentos=5, timeout=240):
    """Llama a chat/completions y devuelve el texto. Reintenta con backoff en 429/5xx."""
    headers = {
        "Authorization": f"Bearer {api_key(proveedor)}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": modelo,
        "messages": mensajes,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    ultimo_error = None
    for intento in range(1, intentos + 1):
        try:
            resp = requests.post(URLS[proveedor], headers=headers, json=payload, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                ultimo_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(min(60, 2 ** intento))
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"] or ""
        except requests.RequestException as exc:
            ultimo_error = str(exc)
            time.sleep(min(60, 2 ** intento))
    raise RuntimeError(f"LLM sin respuesta tras {intentos} intentos: {ultimo_error}")


def extraer_json(texto):
    """Extrae el primer array JSON de una respuesta, tolerando texto alrededor."""
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S).strip()
    texto = re.sub(r"```(?:json)?", "", texto).strip()
    inicio = texto.find("[")
    if inicio == -1:
        raise ValueError(f"Respuesta sin array JSON: {texto[:200]!r}")
    return json.JSONDecoder().raw_decode(texto[inicio:])[0]


def normalizar_etiqueta(valor):
    """Normaliza la etiqueta devuelta por el LLM a una de CLASES (o None)."""
    if not isinstance(valor, str):
        return None
    v = valor.strip().lower().replace("_", " ").replace("-", " ")
    v = (v.replace("á", "a").replace("é", "e").replace("í", "i")
          .replace("ó", "o").replace("ú", "u"))
    equivalencias = {c: c for c in CLASES}
    equivalencias["neutro"] = "neutral"
    return equivalencias.get(v)


def etiquetar_lote(textos, modelo, proveedor="nvidia", ids=None, temperature=0.2,
                   max_tokens=None):
    """Etiqueta una lista de resenas en UNA llamada. Devuelve una lista de dicts
    {id, etiqueta, confianza, justificacion} alineada con `ids` (None si el LLM
    omitio o devolvio una clase invalida para esa resena)."""
    ids = list(ids) if ids is not None else list(range(len(textos)))
    lineas = [
        f'{{"id": {i}, "texto": {json.dumps(str(t), ensure_ascii=False)}}}'
        for i, t in zip(ids, textos)
    ]
    usuario = GUIA_CLASES + "\n\nResenas a etiquetar (un JSON por linea):\n" + "\n".join(lineas)
    if max_tokens is None:
        max_tokens = 200 * len(textos) + 1000
    contenido = completar(
        [{"role": "system", "content": PROMPT_SISTEMA},
         {"role": "user", "content": usuario}],
        modelo=modelo, proveedor=proveedor, max_tokens=max_tokens, temperature=temperature,
    )
    filas = extraer_json(contenido)
    por_id = {}
    for fila in filas:
        if not isinstance(fila, dict):
            continue
        etiqueta = normalizar_etiqueta(fila.get("etiqueta"))
        if etiqueta is None:
            continue
        try:
            confianza = max(0.0, min(1.0, float(fila.get("confianza", 0.5))))
        except (TypeError, ValueError):
            confianza = 0.5
        por_id[fila.get("id")] = {
            "id": fila.get("id"),
            "etiqueta": etiqueta,
            "confianza": round(confianza, 3),
            "justificacion": str(fila.get("justificacion", ""))[:200],
        }
    return [por_id.get(i) for i in ids]
