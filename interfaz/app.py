from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer


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
  --muted: #8a7268;
  --line: #eadcd2;
  --card: #fffaf6;
}
.stApp {
  background: #fbf7f2;
  color: var(--ink);
}
[data-testid="stHeader"] {
  background: transparent;
  height: 1.6rem;
}
[data-testid="stAppViewContainer"] .main .block-container {
  padding: .35rem 1.2rem 1rem;
  max-width: 1240px;
}
.main .block-container {
  padding-top: .35rem !important;
}
[data-testid="stMainBlockContainer"] {
  padding-top: .35rem !important;
}
.element-container:first-child {
  margin-top: 0 !important;
}
[data-testid="stSidebar"] {
  background: #fffaf6;
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] * {
  color: var(--ink) !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: .42rem .55rem;
  border-radius: 12px;
  margin: .1rem 0;
  background: transparent;
}
[data-testid="stSidebar"] [role="radiogroup"] label p {
  display: flex;
  align-items: center;
  gap: .55rem;
}
[data-testid="stSidebar"] [role="radiogroup"] label p::before {
  content: "";
  width: 14px;
  height: 14px;
  flex: 0 0 14px;
  background: #6f7b91;
  display: inline-block;
}
[data-testid="stSidebar"] [role="radiogroup"] label:nth-child(1) p::before {
  -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='black' stroke-width='2'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='m20 20-3.5-3.5'/%3E%3C/svg%3E") center / contain no-repeat;
  mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='black' stroke-width='2'%3E%3Ccircle cx='11' cy='11' r='7'/%3E%3Cpath d='m20 20-3.5-3.5'/%3E%3C/svg%3E") center / contain no-repeat;
}
[data-testid="stSidebar"] [role="radiogroup"] label:nth-child(2) p::before {
  -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='black' stroke-width='2'%3E%3Cpath d='M12 16V8'/%3E%3Cpath d='m8 12 4-4 4 4'/%3E%3Cpath d='M20 16.5A4.5 4.5 0 0 0 15.5 12h-.8A6 6 0 1 0 5 17.2'/%3E%3Cpath d='M8 20h8'/%3E%3C/svg%3E") center / contain no-repeat;
  mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='black' stroke-width='2'%3E%3Cpath d='M12 16V8'/%3E%3Cpath d='m8 12 4-4 4 4'/%3E%3Cpath d='M20 16.5A4.5 4.5 0 0 0 15.5 12h-.8A6 6 0 1 0 5 17.2'/%3E%3Cpath d='M8 20h8'/%3E%3C/svg%3E") center / contain no-repeat;
}
[data-testid="stSidebar"] [role="radiogroup"] label:nth-child(3) p::before {
  -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='black' stroke-width='2'%3E%3Cpath d='M4 19V5'/%3E%3Cpath d='M4 19h16'/%3E%3Crect x='7' y='10' width='3' height='6' rx='1'/%3E%3Crect x='12' y='7' width='3' height='9' rx='1'/%3E%3Crect x='17' y='12' width='3' height='4' rx='1'/%3E%3C/svg%3E") center / contain no-repeat;
  mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='black' stroke-width='2'%3E%3Cpath d='M4 19V5'/%3E%3Cpath d='M4 19h16'/%3E%3Crect x='7' y='10' width='3' height='6' rx='1'/%3E%3Crect x='12' y='7' width='3' height='9' rx='1'/%3E%3Crect x='17' y='12' width='3' height='4' rx='1'/%3E%3C/svg%3E") center / contain no-repeat;
}
[data-testid="stSidebar"] [role="radiogroup"] label:nth-child(4) p::before {
  -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='black' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 11v5'/%3E%3Cpath d='M12 8h.01'/%3E%3C/svg%3E") center / contain no-repeat;
  mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='black' stroke-width='2'%3E%3Ccircle cx='12' cy='12' r='9'/%3E%3Cpath d='M12 11v5'/%3E%3Cpath d='M12 8h.01'/%3E%3C/svg%3E") center / contain no-repeat;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: #f2e7df;
}
h1, h2, h3, h4, h5, h6, p, label, span, div {
  color: var(--ink);
}
textarea, input, [data-baseweb="textarea"] textarea {
  background-color: #ffffff !important;
  color: var(--ink) !important;
  border: 1px solid #d9c4b8 !important;
  border-radius: 14px !important;
}
textarea::placeholder, input::placeholder {
  color: #9b8175 !important;
}
.stButton button {
  background: #a04f3b !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 12px !important;
  height: 2.75rem;
  font-weight: 700;
}
.stButton button * {
  color: #ffffff !important;
}
.stButton button:hover {
  background: #8d432f !important;
  color: #ffffff !important;
}
.stAlert {
  background: #fff3df !important;
  color: var(--ink) !important;
}
.main-shell {
  background: #fbf7f2;
  border: 1px solid var(--line);
  border-radius: 16px;
  min-height: calc(100vh - 3rem);
  padding: 1.35rem 1.45rem;
  box-shadow: 0 18px 60px rgba(0, 0, 0, .18);
}
.hero {
  padding: 1.1rem 1.25rem;
  border-radius: 14px;
  background: #ffffff;
  border: 1px solid var(--line);
  box-shadow: 0 8px 24px rgba(110, 74, 55, 0.05);
}
.hero h1 {
  margin: 0;
  color: var(--accent);
  font-size: 1.65rem;
  letter-spacing: -0.04em;
}
.hero p {
  margin: .35rem 0 0;
  color: var(--muted);
  font-size: .88rem;
  max-width: 780px;
}
.block-spacer {
  height: .25rem;
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
.section-title {
  font-size: 1.05rem;
  font-weight: 800;
  margin: .1rem 0 .25rem;
}
.section-caption {
  color: var(--muted);
  font-size: .82rem;
  margin-bottom: .45rem;
}
.stMarkdown code {
  color: #2f7d57 !important;
  background: #f7dfd5 !important;
}
.result-box {
  padding: 1rem;
  border-radius: 14px;
  background: #ffffff;
  border: 1px solid var(--line);
}
.result-box h3 {
  margin: .8rem 0 .45rem;
  font-size: 1.55rem;
}
.result-box p {
  font-size: .86rem;
  line-height: 1.45;
  margin-bottom: 0;
}
.confidence-line {
  height: 8px;
  background: #eee7e0;
  border-radius: 999px;
  overflow: hidden;
  margin: .4rem 0 .9rem;
}
.confidence-fill {
  height: 100%;
  background: #2f7d57;
  border-radius: 999px;
}
.prob-title {
  color: var(--muted);
  font-size: .68rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  font-weight: 800;
  margin: .95rem 0 .5rem;
}
.prob-row {
  display: grid;
  grid-template-columns: 86px 1fr 44px;
  align-items: center;
  gap: .55rem;
  margin: .32rem 0;
  font-size: .72rem;
}
.prob-name {
  color: #6f7b91;
  text-align: right;
}
.prob-track {
  height: 10px;
  border-radius: 999px;
  background: #f0f1f3;
  overflow: hidden;
}
.prob-fill {
  height: 100%;
  border-radius: 999px;
}
.prob-value {
  color: var(--muted);
  font-variant-numeric: tabular-nums;
}
.upload-card, .preview-card, .info-card {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1rem;
  box-shadow: 0 8px 24px rgba(110, 74, 55, 0.04);
}
.metric-tile {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1rem;
  min-height: 128px;
  box-shadow: 0 8px 24px rgba(110, 74, 55, 0.04);
}
.metric-tile .label {
  font-size: .76rem;
  color: var(--muted);
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .04em;
}
.metric-tile .value {
  font-size: 1.65rem;
  color: var(--accent);
  font-weight: 900;
  margin: .55rem 0 .2rem;
}
.metric-ring {
  height: 8px;
  border-radius: 999px;
  background: #eee7e0;
  overflow: hidden;
  margin-top: .65rem;
}
.metric-ring div {
  height: 100%;
  background: var(--accent);
  border-radius: 999px;
}
.table-card {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1rem;
  box-shadow: 0 8px 24px rgba(110, 74, 55, 0.04);
}
.class-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0 .45rem;
  font-size: .84rem;
}
.class-table th {
  color: var(--muted);
  font-size: .68rem;
  text-transform: uppercase;
  letter-spacing: .04em;
  text-align: left;
  padding: .15rem .55rem;
}
.class-table td {
  padding: .2rem .55rem;
}
.score-pill {
  display: inline-block;
  min-width: 48px;
  text-align: center;
  padding: .22rem .45rem;
  border-radius: 5px;
  color: #ffffff;
  font-weight: 800;
}
.score-high { background: #a04f3b; }
.score-mid { background: #c99484; }
.score-low { background: #ead0c5; color: var(--ink); }
.support-cell {
  color: #5f514b;
  font-variant-numeric: tabular-nums;
}
.table-note {
  margin-top: .7rem;
  padding: .75rem .85rem;
  border-radius: 12px;
  background: #fff4ee;
  color: var(--muted);
  font-size: .82rem;
  line-height: 1.45;
}
.table-note strong {
  color: var(--accent);
}
.model-note {
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1rem;
  height: 100%;
  box-shadow: 0 8px 24px rgba(110, 74, 55, 0.04);
}
.workflow-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: .85rem;
}
.workflow-step {
  background: #fffaf6;
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: .85rem;
}
.workflow-step strong {
  color: var(--accent);
}
.workflow-step p {
  color: var(--muted);
  font-size: .82rem;
  margin: .3rem 0 0;
}
.model-info-line {
  border-top: 1px solid var(--line);
  padding-top: .75rem;
  margin-top: .75rem;
  font-size: .84rem;
}
.model-info-label {
  color: var(--muted);
  font-size: .7rem;
  text-transform: uppercase;
  letter-spacing: .05em;
  font-weight: 800;
}
.model-note code {
  white-space: normal !important;
  overflow-wrap: anywhere;
  word-break: break-word;
}
.empty-preview {
  display: grid;
  place-items: center;
  min-height: 210px;
  border-radius: 14px;
  background: #fbf2eb;
  color: var(--muted);
  text-align: center;
  font-size: .86rem;
}
[data-testid="stVerticalBlock"] {
  gap: .55rem;
}
.loading-card {
  max-width: 820px;
  margin: 12vh auto 0;
  padding: 2rem;
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.76);
  border: 1px solid #eadcd2;
  box-shadow: 0 24px 60px rgba(110, 74, 55, 0.12);
  text-align: center;
}
.loading-card h1 {
  color: var(--accent);
  margin-bottom: .4rem;
}
.loading-card p {
  color: #7a6257;
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


def precargar_modelo():
    pantalla = st.empty()
    with pantalla.container():
        st.markdown(
            """
            <div class="loading-card">
              <h1>Preparando Sentimix Peru</h1>
              <p>Cargando el modelo BETO entrenado. La primera carga puede tardar unos segundos.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.spinner("Inicializando tokenizador y pesos del modelo..."):
            cargar_modelo()
    pantalla.empty()


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
        height=220,
        showlegend=False,
        margin=dict(l=10, r=16, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickformat=".0%",
        yaxis_title="",
        xaxis_title="",
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


def html_probabilidades(pred):
    filas = []
    for clase in CLASES:
        prob = float(pred[f"prob_{clase}"])
        color = COLORES[clase]
        filas.append(
            f"<div class='prob-row'>"
            f"<div class='prob-name'>{clase}</div>"
            f"<div class='prob-track'><div class='prob-fill' style='width:{prob * 100:.1f}%; background:{color};'></div></div>"
            f"<div class='prob-value'>{prob:.0%}</div>"
            f"</div>"
        )
    return "<div class='prob-title'>Distribucion de probabilidades por clase</div>" + "".join(filas)


def tarjeta_resultado(pred):
    sentimiento = pred["sentimiento"]
    confianza = float(pred["confianza"])
    if confianza >= 0.75:
        fuerza = "alta"
    elif confianza >= 0.55:
        fuerza = "media"
    else:
        fuerza = "baja"
    st.markdown(
        f"<div class='result-box'>"
        f"<span class='badge'>{sentimiento}</span>"
        f"<h3>Confianza: {confianza:.1%}</h3>"
        f"<div class='confidence-line'><div class='confidence-fill' style='width:{confianza * 100:.1f}%;'></div></div>"
        f"<p>El modelo clasifica la resena como <strong>{sentimiento}</strong> con confianza {fuerza}. "
        f"Si la confianza es baja, conviene revisar el texto manualmente porque puede contener matices mixtos.</p>"
        f"{html_probabilidades(pred)}"
        f"</div>",
        unsafe_allow_html=True,
    )


precargar_modelo()


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

st.markdown("<div class='block-spacer'></div>", unsafe_allow_html=True)

if "Analizar resena" in seccion:
    col_texto, col_resultado = st.columns([1.25, 1], gap="large")
    with col_texto:
        st.markdown("<div class='section-title'>Espacio de analisis</div>", unsafe_allow_html=True)
        ejemplo = "La atencion fue rapida y el personal fue muy amable. Lo recomiendo."
        texto = st.text_area(
            "Escribe una resena",
            value=ejemplo,
            height=150,
            max_chars=5000,
            placeholder="Ejemplo: El servicio fue lento, pero el personal intento ayudar...",
        )
        analizar = st.button("Analizar sentimiento", type="primary", use_container_width=True)

    with col_resultado:
        st.markdown("<div class='section-title'>Resultado</div>", unsafe_allow_html=True)
        if analizar and texto.strip():
            pred = predecir([texto.strip()]).iloc[0]
            tarjeta_resultado(pred)
        else:
            st.markdown(
                "<div class='result-box'>"
                "<span class='badge'>pendiente</span>"
                "<h3>Sin prediccion</h3>"
                "<p>Ingresa una resena y presiona el boton para obtener el sentimiento y las probabilidades.</p>"
                "</div>",
                unsafe_allow_html=True,
            )

elif "Analisis por lote" in seccion:
    st.markdown("<div class='section-title'>Analisis por lote</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-caption'>Procesa multiples resenas subiendo un archivo CSV.</div>", unsafe_allow_html=True)
    col_upload, col_preview = st.columns([.8, 1.6], gap="large")
    with col_upload:
        with st.container(border=True):
            st.markdown("**Subir datos**")
            archivo = st.file_uploader("Subir archivo CSV", type=["csv"], label_visibility="collapsed")
            st.caption("Formato esperado: una columna con resenas o `comentario_limpio`.")

    with col_preview:
        with st.container(border=True):
            st.markdown("**Previsualizacion de datos**")
            if archivo is None:
                st.markdown(
                    "<div class='empty-preview'>Sube un archivo para visualizar las primeras filas.</div>",
                    unsafe_allow_html=True,
                )
            else:
                df = pd.read_csv(archivo).fillna("")
                if df.empty or len(df.columns) == 0:
                    st.warning("El CSV no contiene filas para analizar.")
                else:
                    st.dataframe(df.head(8), use_container_width=True)
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

elif "Metricas del modelo" in seccion:
    st.markdown("<div class='section-title'>Metricas del modelo final</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-caption'>Evaluacion del modelo BETO final sobre el 20% reservado para test.</div>",
        unsafe_allow_html=True,
    )
    metricas = [
        ("F1-macro", 0.6688),
        ("Accuracy", 0.6760),
        ("Balanced accuracy", 0.6756),
        ("F1 weighted", 0.6719),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, metricas):
        with col:
            st.markdown(
                f"<div class='metric-tile'>"
                f"<div class='label'>{label}</div>"
                f"<div class='value'>{value:.1%}</div>"
                f"<div class='section-caption'>{value:.4f}</div>"
                f"<div class='metric-ring'><div style='width:{value * 100:.1f}%'></div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.write("")
    col_tabla, col_modelo = st.columns([1.55, .9], gap="large")
    with col_tabla:
        with st.container(border=True):
            st.markdown("**Performance por Clase**")
            filas_clase = [
                ("Muy Negativo", 0.86, 0.88, 0.87, 186),
                ("Negativo", 0.73, 0.67, 0.70, 165),
                ("Neutral", 0.58, 0.43, 0.49, 139),
                ("Positivo", 0.68, 0.58, 0.62, 290),
                ("Muy Positivo", 0.55, 0.82, 0.66, 180),
            ]

            def score_class(valor):
                if valor >= 0.75:
                    return "score-high"
                if valor >= 0.60:
                    return "score-mid"
                return "score-low"

            filas_html = "".join(
                f"<tr>"
                f"<td>{clase}</td>"
                f"<td><span class='score-pill {score_class(precision)}'>{precision:.2f}</span></td>"
                f"<td><span class='score-pill {score_class(recall)}'>{recall:.2f}</span></td>"
                f"<td><span class='score-pill {score_class(f1)}'>{f1:.2f}</span></td>"
                f"<td class='support-cell'>{support:,}</td>"
                f"</tr>"
                for clase, precision, recall, f1, support in filas_clase
            )
            st.markdown(
                "<table class='class-table'>"
                "<thead><tr><th>Clase</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead>"
                f"<tbody>{filas_html}</tbody>"
                "</table>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='table-note'>"
                "<strong>Ejemplo:</strong> en test habia 180 resenas realmente <strong>muy positivo</strong>. "
                "El modelo recupero el 82% de ellas (recall), pero no todo lo que predijo como muy positivo era correcto, "
                "por eso su precision es 55%. <strong>Support</strong> es solo la cantidad real de ejemplos de esa clase."
                "</div>",
                unsafe_allow_html=True,
            )

    with col_modelo:
        st.markdown(
            "<div class='model-note'>"
            "<span class='badge'>Seleccion de modelo</span>"
            "<h3>BETO final</h3>"
            "<p>BETO fue seleccionado por su mejor F1-macro en validacion y luego evaluado en el conjunto test reservado.</p>"
            "<div class='model-info-line'><div class='model-info-label'>Metrica principal</div>F1-macro, adecuada para clases desbalanceadas.</div>"
            "<div class='model-info-line'><div class='model-info-label'>Clases</div>muy negativo, negativo, neutral, positivo y muy positivo.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

else:
    st.markdown("<div class='section-title'>Acerca del proyecto</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-caption'>Sentimix Peru clasifica resenas de consumidores peruanos usando el modelo BETO final del flujo oficial.</div>",
        unsafe_allow_html=True,
    )
    col_flujo, col_info = st.columns([1.6, .8], gap="large")
    with col_flujo:
        with st.container(border=True):
            st.markdown("**Metodologia del flujo**")
            st.markdown(
                "<div class='workflow-grid'>"
                "<div class='workflow-step'><strong>1. Scraping</strong><p>Recoleccion de resenas de consumidores peruanos desde fuentes publicas.</p></div>"
                "<div class='workflow-step'><strong>2. Limpieza</strong><p>Normalizacion textual y preparacion del dataset procesado.</p></div>"
                "<div class='workflow-step'><strong>3. Split 80/20</strong><p>Separacion entre desarrollo y test reservado; del desarrollo se extrae la semilla.</p></div>"
                "<div class='workflow-step'><strong>4. Semilla manual</strong><p>Etiquetado manual inicial de 500 registros para entrenar y comparar modelos.</p></div>"
                "<div class='workflow-step'><strong>5. Validacion cruzada</strong><p>Comparacion entre SVM, Naive Bayes, CNN, LSTM, BETO y XLM-RoBERTa.</p></div>"
                "<div class='workflow-step'><strong>6. BETO inicial</strong><p>El mejor modelo etiqueta el resto del conjunto de desarrollo.</p></div>"
                "<div class='workflow-step'><strong>7. Clustering y revision</strong><p>Revision de coherencia por clusters y casos dudosos antes de consolidar etiquetas.</p></div>"
                "<div class='workflow-step'><strong>8. Entrenamiento final</strong><p>BETO se entrena con el conjunto consolidado y se evalua en test reservado.</p></div>"
                "</div>",
                unsafe_allow_html=True,
            )

    with col_info:
        st.markdown(
            f"<div class='model-note'>"
            f"<span class='badge'>Modelo final</span>"
            f"<h3>BETO sentimiento</h3>"
            f"<div class='model-info-line'><div class='model-info-label'>Ruta principal</div><code>{MODEL_DIR}</code></div>"
            f"<div class='model-info-line'><div class='model-info-label'>Arquitectura</div>Transformer BETO para clasificacion de secuencias.</div>"
            f"<div class='model-info-line'><div class='model-info-label'>Salida</div>5 clases de sentimiento.</div>"
            f"<div class='model-info-line'><div class='model-info-label'>Uso</div>Prediccion individual y analisis por lote desde CSV.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
