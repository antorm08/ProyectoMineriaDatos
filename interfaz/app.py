from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "modelo_entrenado" / "beto_sentimiento"
CLASES = ["muy negativo", "negativo", "neutral", "positivo", "muy positivo"]
COLORES = {
    "muy negativo": "#8f2d1f",
    "negativo": "#c55a3c",
    "neutral": "#7f8c8d",
    "positivo": "#5d9c72",
    "muy positivo": "#2f7d57",
}


st.set_page_config(
    page_title="Sentimix Peru",
    page_icon="SP",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
:root {
  --accent: #a04f3b;
  --accent-soft: #f3c0ad;
  --paper: #fbf7f2;
  --ink: #3b2b25;
}
.stApp {
  background: linear-gradient(120deg, #f8f2ea 0%, #fffaf6 45%, #f4eee7 100%);
  color: var(--ink);
}
[data-testid="stSidebar"] {
  background: rgba(255, 250, 245, 0.92);
  border-right: 1px solid #eadcd2;
}
.hero {
  padding: 1.4rem 1.8rem 1rem;
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.58);
  box-shadow: 0 24px 60px rgba(110, 74, 55, 0.10);
  border: 1px solid rgba(160, 79, 59, 0.12);
}
.hero h1 {
  margin: 0;
  color: var(--accent);
  font-size: 2.4rem;
  letter-spacing: -0.04em;
}
.hero p {
  margin-top: 0.55rem;
  color: #7a6257;
  max-width: 780px;
}
.metric-card {
  padding: 1.1rem 1.2rem;
  border-radius: 22px;
  background: rgba(255,255,255,.72);
  border: 1px solid #eadcd2;
  box-shadow: 0 15px 35px rgba(110, 74, 55, 0.08);
}
.metric-label { color: #8a7268; font-size: .86rem; }
.metric-value { color: var(--accent); font-size: 1.65rem; font-weight: 800; }
.badge {
  display: inline-block;
  padding: .35rem .65rem;
  border-radius: 999px;
  background: #f7dfd5;
  color: #8d432f;
  font-weight: 700;
  font-size: .8rem;
}
.result-box {
  padding: 1.2rem;
  border-radius: 24px;
  background: #fffaf6;
  border: 1px solid #eadcd2;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando modelo BETO entrenado...")
def cargar_modelo():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"No existe el modelo entrenado en {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return tokenizer, model, device, id2label


def predecir(textos, batch_size=16):
    tokenizer, model, device, id2label = cargar_modelo()
    filas = []
    with torch.no_grad():
        for i in range(0, len(textos), batch_size):
            lote = textos[i:i + batch_size]
            enc = tokenizer(
                lote,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            for texto, prob in zip(lote, probs):
                pred_idx = int(prob.argmax())
                etiqueta = id2label.get(pred_idx, CLASES[pred_idx])
                fila = {
                    "texto": texto,
                    "sentimiento": etiqueta,
                    "confianza": float(prob[pred_idx]),
                }
                for idx, clase in enumerate(CLASES):
                    fila[f"prob_{clase}"] = float(prob[idx])
                filas.append(fila)
    return pd.DataFrame(filas)


def grafico_probabilidades(fila):
    datos = pd.DataFrame({
        "clase": CLASES,
        "probabilidad": [fila[f"prob_{clase}"] for clase in CLASES],
    })
    fig = px.bar(
        datos,
        x="probabilidad",
        y="clase",
        orientation="h",
        color="clase",
        color_discrete_map=COLORES,
        range_x=[0, 1],
    )
    fig.update_layout(
        height=310,
        showlegend=False,
        margin=dict(l=10, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickformat=".0%",
        yaxis_title="",
        xaxis_title="Probabilidad",
    )
    return fig


def interpretacion(sentimiento, confianza):
    if confianza >= 0.75:
        fuerza = "alta"
    elif confianza >= 0.55:
        fuerza = "media"
    else:
        fuerza = "baja"
    return (
        f"El modelo clasifica la resena como **{sentimiento}** con confianza {fuerza}. "
        "Si la confianza es baja, conviene revisar el texto manualmente porque puede contener matices mixtos."
    )


with st.sidebar:
    st.markdown("# Sentimix Peru")
    st.caption("Clasificador BETO para resenas de consumidores peruanos")
    seccion = st.radio(
        "Navegacion",
        ["Analizar resena", "Analisis por lote", "Metricas del modelo", "Acerca del proyecto"],
    )
    st.divider()
    st.markdown("**Modelo final**")
    st.markdown("`BETO` fine-tuned")
    st.markdown("<span class='badge'>5 clases de sentimiento</span>", unsafe_allow_html=True)


st.markdown(
    """
    <div class="hero">
      <h1>Analizador de Sentimiento Multiclase</h1>
      <p>Evalua resenas de consumidores peruanos con el modelo BETO entrenado en el flujo final del proyecto.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

if seccion == "Analizar resena":
    col_texto, col_resultado = st.columns([1.25, 1], gap="large")
    with col_texto:
        st.subheader("Espacio de analisis")
        ejemplo = "La atencion fue rapida y el personal fue muy amable. Lo recomiendo."
        texto = st.text_area(
            "Escribe una resena",
            value=ejemplo,
            height=210,
            max_chars=5000,
            placeholder="Ejemplo: El servicio fue lento, pero el personal intento ayudar...",
        )
        analizar = st.button("Analizar sentimiento", type="primary", use_container_width=True)

    with col_resultado:
        st.subheader("Resultado")
        if analizar and texto.strip():
            pred = predecir([texto.strip()]).iloc[0]
            sentimiento = pred["sentimiento"]
            confianza = pred["confianza"]
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"<span class='badge'>{sentimiento}</span>", unsafe_allow_html=True)
            st.markdown(f"### Confianza: {confianza:.1%}")
            st.markdown(interpretacion(sentimiento, confianza))
            st.markdown("</div>", unsafe_allow_html=True)
            st.plotly_chart(grafico_probabilidades(pred), use_container_width=True)
        else:
            st.info("Ingresa una resena y presiona el boton para obtener la prediccion.")

elif seccion == "Analisis por lote":
    st.subheader("Analisis por lote")
    st.write("Sube un CSV con una columna de texto. Si existe `comentario_limpio`, se usara automaticamente.")
    archivo = st.file_uploader("CSV de resenas", type=["csv"])
    if archivo is not None:
        df = pd.read_csv(archivo).fillna("")
        if df.empty or len(df.columns) == 0:
            st.warning("El CSV no contiene filas para analizar.")
        else:
            columna_default = "comentario_limpio" if "comentario_limpio" in df.columns else df.columns[0]
            columna = st.selectbox("Columna de texto", list(df.columns), index=list(df.columns).index(columna_default))
            max_filas = min(1000, len(df))
            limite = st.slider("Maximo de filas a analizar", min_value=1, max_value=max_filas, value=min(100, max_filas))
            if st.button("Analizar lote", type="primary"):
                textos = df[columna].astype(str).head(limite).tolist()
                salida = predecir(textos)
                resultado = pd.concat([df.head(limite).reset_index(drop=True), salida.drop(columns=["texto"])], axis=1)
                st.dataframe(resultado, use_container_width=True)
                dist = resultado["sentimiento"].value_counts().rename_axis("sentimiento").reset_index(name="cantidad")
                fig = px.pie(dist, names="sentimiento", values="cantidad", color="sentimiento", color_discrete_map=COLORES)
                st.plotly_chart(fig, use_container_width=True)
                st.download_button(
                    "Descargar resultados",
                    resultado.to_csv(index=False).encode("utf-8-sig"),
                    file_name="predicciones_sentimiento.csv",
                    mime="text/csv",
                )

elif seccion == "Metricas del modelo":
    st.subheader("Metricas del modelo final")
    c1, c2, c3, c4 = st.columns(4)
    metricas = [
        ("F1-macro", "0.6688"),
        ("Accuracy", "0.6760"),
        ("Balanced accuracy", "0.6756"),
        ("F1 weighted", "0.6719"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4], metricas):
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
                unsafe_allow_html=True,
            )
    st.write("")
    st.markdown(
        """
        El modelo final es BETO, seleccionado despues de comparar SVM, Naive Bayes, CNN, LSTM, BETO y XLM-RoBERTa.
        La metrica principal fue F1-macro por tratarse de un problema multiclase con clases desbalanceadas.
        """
    )
    reporte = PROJECT_ROOT / "reports" / "14_entrenamiento_final" / "reporte_clasificacion_beto_final_test.txt"
    if reporte.exists():
        st.code(reporte.read_text(encoding="utf-8"), language="text")

else:
    st.subheader("Acerca del proyecto")
    st.markdown(
        """
        Este prototipo usa el modelo ganador del flujo final del proyecto para clasificar resenas en cinco clases:
        `muy negativo`, `negativo`, `neutral`, `positivo` y `muy positivo`.

        Flujo resumido:

        1. Scraping y limpieza de resenas.
        2. Semilla manual de 500 registros.
        3. Validacion cruzada con seis modelos.
        4. Seleccion de BETO como modelo inicial.
        5. Etiquetado del resto del 80% de desarrollo.
        6. Revision con clustering.
        7. Entrenamiento final y evaluacion en test reservado.
        """
    )
    st.info(f"Ruta del modelo: {MODEL_DIR}")
