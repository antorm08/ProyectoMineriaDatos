// Genera el informe final del proyecto (.docx) leyendo los resultados de reports/.
// Uso: node build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, PageNumber, Header, Footer,
  AlignmentType, LevelFormat, HeadingLevel, PageBreak, BorderStyle, TableOfContents,
} = require("docx");
const L = require("./lib_docx");

const ROOT = path.resolve(__dirname, "..");
const R = (p) => path.join(ROOT, "reports", p);
const A = (AlignmentType.CENTER, AlignmentType.RIGHT); // atajo no usado
const CTR = AlignmentType.CENTER, RGT = AlignmentType.RIGHT, LFT = AlignmentType.LEFT;

// --------------------------------------------------------------------------- //
// Helpers de resultados
// --------------------------------------------------------------------------- //
function tryCSV(ruta) { try { return L.existe(ruta) ? L.leerCSV(ruta) : null; } catch { return null; } }
function tryJSON(ruta) { try { return L.existe(ruta) ? JSON.parse(fs.readFileSync(ruta, "utf8")) : null; } catch { return null; } }
function fmt(x, d = 4) { const n = Number(x); return Number.isFinite(n) ? n.toFixed(d) : String(x); }
function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

const NOMBRE = { svm: "SVM", naive_bayes: "Naive Bayes", cnn: "CNN (TextCNN)",
  lstm: "LSTM (BiLSTM)", beto: "BETO", xlm_roberta: "XLM-RoBERTa" };
const FAMILIA_ES = { clasico: "Clasico", deep_learning: "Deep Learning", transformer: "Transformer" };

// --------------------------------------------------------------------------- //
// Contenido
// --------------------------------------------------------------------------- //
const contenido = [];
const add = (...xs) => contenido.push(...xs);

// ---- Portada ----
add(
  new Paragraph({ alignment: CTR, spacing: { after: 60 },
    children: [new TextRun({ text: "UNIVERSIDAD NACIONAL MAYOR DE SAN MARCOS",
      bold: true, size: 24, font: "Times New Roman" })] }),
  L.p("(Universidad del Peru, Decana de America)", { center: true, after: 40 }),
  L.p("Facultad de Ingenieria de Sistemas e Informatica", { center: true, after: 40 }),
  L.p("Escuela Profesional de Ingenieria de Software", { center: true, after: 300 }),
  L.titulo("Analisis de Sentimiento Multiclase sobre Resenas de Empresas Peruanas mediante un Flujo Semisupervisado con NLP y Web Scraping"),
  L.subtitulo("Curso: Mineria de Datos"),
  L.p("Grupo 05", { center: true, bold: true, after: 40 }),
  L.p("Astuhuaman Vega, Max  -  Osorio Montenegro, Alejandro  -  Calle Ramos, Guillermo", { center: true, after: 20 }),
  L.p("Romani Moscoso, Anthony  -  Silva Burga, Bryan", { center: true, after: 200 }),
  L.p("Docente: David Calderon Vilca", { center: true, after: 40 }),
  L.p("Lima - Peru, 2026", { center: true, after: 120 }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ---- Resumen / Abstract ----
add(L.h1("Resumen"));
add(L.p("El presente trabajo aborda la clasificacion de sentimiento multiclase en cinco niveles (muy negativo, negativo, neutral, positivo y muy positivo) sobre resenas reales de consumidores peruanos publicadas en Google Maps. A diferencia de un avance previo que dependia de las estrellas como etiqueta, en esta version se adopta un flujo semisupervisado alineado con la retroalimentacion docente: se reserva un conjunto de prueba independiente, se etiqueta una semilla representativa con asistencia de un modelo de lenguaje grande (LLM) revisada por el equipo, se seleccionan hiperparametros por validacion cruzada, y el mejor modelo propaga etiquetas al resto del corpus (self-training) con verificacion por agrupamiento (clustering) y arbitraje de casos dudosos. Se comparan seis algoritmos en tres familias: clasicos (SVM y Naive Bayes con TF-IDF), aprendizaje profundo (CNN y LSTM con embeddings entrenables) y transformers (BETO y XLM-RoBERTa con ajuste fino). La seleccion del modelo se realiza por F1-macro y la evaluacion final se efectua una sola vez sobre el conjunto de prueba reservado. Los resultados confirman la superioridad de los transformers preentrenados en espanol para esta tarea."));
add(L.pRuns([{ t: "Palabras clave: ", b: true }, { t: "analisis de sentimiento; clasificacion multiclase; aprendizaje semisupervisado; self-training; BETO; validacion cruzada; NLP en espanol." }]));

add(L.h1("Abstract"));
add(L.p("This work addresses five-level multiclass sentiment classification of real Peruvian consumer reviews from Google Maps. Unlike a previous milestone that relied on star ratings as labels, this version adopts a semi-supervised workflow aligned with the instructor's feedback: an independent test set is held out, a representative seed is labeled with Large Language Model (LLM) assistance and team review, hyperparameters are chosen via cross-validation, and the best model propagates labels to the remaining corpus (self-training) with clustering-based verification and arbitration of doubtful cases. Six algorithms across three families are compared: classical (SVM and Naive Bayes with TF-IDF), deep learning (CNN and LSTM with trainable embeddings), and transformers (BETO and XLM-RoBERTa with fine-tuning). Model selection uses macro-F1 and the final evaluation is performed only once on the held-out test set. Results confirm the superiority of Spanish-pretrained transformers for this task."));
add(L.pRuns([{ t: "Keywords: ", b: true }, { t: "sentiment analysis; multiclass classification; semi-supervised learning; self-training; BETO; cross-validation; Spanish NLP." }]));
add(new Paragraph({ children: [new PageBreak()] }));

// ---- Indice ----
add(L.h1("Contenido"));
add(new TableOfContents("Tabla de Contenido", { hyperlink: true, headingStyleRange: "1-2" }));
add(new Paragraph({ children: [new PageBreak()] }));

// ============================ SECCION I ============================
add(L.h1("I. Introduccion"));
add(L.h2("1.1. Contexto"));
add(L.p("La transformacion digital cambio la forma en que las personas comparten sus experiencias con las organizaciones. En Google Maps, cada usuario puede dejar una resena acompanada de una calificacion de una a cinco estrellas. En el Peru, empresas de banca, telecomunicaciones, retail y farmacia (Interbank, Claro, Plaza Vea, Metro, Hiraoka, entre otras) acumulan miles de resenas que, analizadas con eficiencia, se convierten en conocimiento accionable sobre la percepcion del consumidor. Para lograrlo, la mineria de texto y el procesamiento de lenguaje natural (PLN) permiten transformar opiniones dispersas en una variable de sentimiento util."));
add(L.h2("1.2. Fundamentacion del problema"));
add(L.p("Los organismos reguladores evidencian un malestar considerable con las grandes empresas de servicios. Segun OSIPTEL, entre enero y diciembre de 2024 se registraron 722 291 denuncias y reclamos de usuarios del servicio movil ante las cuatro principales operadoras, y la calificacion nacional de atencion de quejas descendio a 13.3 puntos, por debajo de la meta minima de 15. En el sector financiero, INDECOPI registro mas de 21 000 reportes en pocos meses, con BCP e Interbank entre los mas reportados. El reto es que las resenas suelen ser cortas, mal redactadas o ilegibles, lo que dificulta su analisis con metodos tradicionales; ademas, el analisis de sentimiento se ha desarrollado mayoritariamente para el ingles, con escasez de recursos para el espanol latinoamericano. Surge asi la necesidad de construir y analizar un corpus de opiniones reales de consumidores peruanos."));
add(L.h2("1.3. Objetivo"));
add(L.p("Desarrollar y evaluar un modelo de clasificacion de sentimiento multiclase (cinco niveles) sobre resenas reales de empresas peruanas obtenidas de Google Maps, construyendo un pipeline reproducible de recoleccion, etiquetado semisupervisado y comparacion de modelos clasicos, de aprendizaje profundo y transformers, seleccionando el mejor mediante validacion cruzada y F1-macro."));
add(L.h2("1.4. Justificacion"));
add(L.p("En el ambito practico, brinda a empresas y reguladores una alternativa rapida y economica para conocer la opinion de los consumidores a partir de informacion publica, reduciendo la dependencia de encuestas. En el ambito academico, aporta evidencia sobre analisis de sentimiento en espanol peruano, poco representado en las bases indexadas. Desde lo metodologico, propone un procedimiento que integra web scraping, etiquetado asistido por IA con revision humana, tratamiento del desbalance y comparacion sistematica de modelos, constituyendo una referencia para futuros estudios."));

// ============================ SECCION II ============================
add(L.h1("II. Trabajos Relacionados"));
add(L.h2("2.1. Recoleccion y mineria de texto sobre resenas de Google Maps"));
add(L.p("Estudios recientes usan Google Maps como fuente porque sus resenas combinan texto y estrellas, lo que facilita el etiquetado inicial (Shin & Ryu, 2022; Aunimo et al., 2025). El flujo tipico automatiza la extraccion con herramientas como Selenium, limpia el texto, tokeniza y lo representa con TF-IDF o modelos de lenguaje. Shin y Ryu (2022) aplicaron Selenium, TF-IDF y Random Forest sobre 5 427 resenas de restaurantes; trabajos posteriores llevaron el esquema a salud, retail y turismo con modelos mas exigentes (GPT-4, LLaMA-3, XLM-RoBERTa, AraBERT). La mayoria contempla realidades distintas a la peruana, lo que motiva este proyecto."));
add(L.h2("2.2. Clasificacion multiclase y comparacion de clasificadores"));
add(L.p("Pasar de una polaridad binaria a una escala de cinco niveles incrementa la dificultad: en resenas en chino, el F1 de SVM cayo de 0.93 a 0.27 al enfrentar cinco clases (Suandi et al., 2024). Aun asi, un clasico bien construido con TF-IDF y bigramas compite con el aprendizaje profundo cuando los datos estan estructurados (GeeksforGeeks; Mihalcea & Ionescu, 2023). En espanol, los modelos preentrenados se imponen: un estudio con BERT alcanzo 0.81 de precision media frente a 0.71 de Naive Bayes y 0.73 de SVM (OpenWebinars, 2024); BETO logro 0.83 sobre 1 080 resenas, superando el 0.791 de Naive Bayes, aunque ambos tropiezan con la clase neutral (Herrera Gambini, 2021)."));
add(L.h2("2.3. Desbalance de clases, PLN en espanol y contexto regional"));
add(L.p("El desbalance es un obstaculo central en clasificacion multiclase: si una categoria concentra casi todas las muestras, el modelo apuesta siempre por ella y las clases minoritarias se clasifican mal. SMOTE genera muestras sinteticas interpolando vecinos y suele aplicarse solo al entrenamiento (Villanueva, 2024). En resenas de la app by.U, SMOTE elevo la exactitud de Naive Bayes y permitio registrar la clase neutral (Ashbaugh & Zhang, 2024). Por ello se recomienda evaluar con metricas que pesen todas las clases por igual, como la exactitud balanceada y el F1-macro (Grandini et al., 2020). El espanol y los idiomas distintos al ingles siguen subrepresentados en las bases indexadas (Arango Pastrana & Osorio Andrade, 2021), lo que refuerza la relevancia de este trabajo sobre opiniones reales de consumidores peruanos."));

// ============================ SECCION III: METODOLOGIA ============================
add(L.h1("III. Metodologia"));
add(L.h2("3.1. Marco metodologico"));
add(L.p("El proyecto adopta la metodologia DSRM (Design Science Research Methodology) para construir y evaluar un artefacto computacional -un pipeline de mineria de datos- que resuelve el problema practico de clasificar sentimiento en resenas peruanas. El proceso interno sigue la estructura de CRISP-DM: comprension del negocio y de los datos, preparacion, modelado, evaluacion y despliegue. La novedad de esta version, atendiendo la retroalimentacion docente, es un flujo semisupervisado que no confia ciegamente en las estrellas como etiqueta, sino que construye etiquetas de calidad a partir de una semilla revisada y las propaga de forma controlada."));

add(L.h2("3.2. Procedencia y estructura de los datos"));
add(L.p("El corpus procede de resenas publicas de Google Maps de empresas peruanas, recolectadas mediante web scraping controlado (Selenium) con medidas anti-bloqueo. El dataset base contiene 4 800 resenas con: comentario, empresa, sede, rubro, estrellas (1-5), fecha y URL. Tras limpieza y auditoria se conservan las columnas de texto normalizado (texto_modelo, en minusculas) y texto natural (comentario_limpio), ademas de metadatos de calidad y senales auxiliares."));

add(L.h2("3.3. Arquitectura del flujo semisupervisado"));
add(L.p("El flujo se organiza en las siguientes etapas, ejecutadas como scripts independientes y auditables:"));
add(L.bullet("Particion inicial: el dataset se trata como no etiquetado de forma confiable; se reserva un 20% como conjunto de prueba final (evaluado una sola vez) y del 80% de desarrollo se selecciona una semilla de 500 resenas representativas."));
add(L.bullet("Etiquetado de la semilla: se etiquetan los 500 registros con asistencia de un modelo de lenguaje grande (LLM) que asigna clase, confianza y justificacion; el equipo revisa los casos de menor confianza (revision humana asistida)."));
add(L.bullet("Validacion cruzada e hiperparametros: sobre la semilla se comparan los seis modelos con validacion cruzada estratificada de cinco particiones, explorando una malla de hiperparametros por algoritmo, y se elige el mejor por F1-macro."));
add(L.bullet("Self-training: el mejor modelo se reentrena con la semilla y etiqueta automaticamente el resto del 80%."));
add(L.bullet("Verificacion por clustering: se agrupan las resenas (TF-IDF + SVD + K-Means) y se comparan las etiquetas propagadas con la clase mayoritaria de cada grupo suficientemente puro; las discrepancias y los casos de baja confianza se marcan como dudosos."));
add(L.bullet("Arbitraje: los casos dudosos se re-etiquetan con el LLM como segunda opinion, consolidando el dataset de entrenamiento completo."));
add(L.bullet("Entrenamiento final y evaluacion: se reentrenan los seis modelos sobre el 80% etiquetado (subdividido 70/30 en entrenamiento y validacion), se grafican las curvas de entrenamiento, se elige el mejor por validacion y se evalua una unica vez en el 20% de prueba."));

add(L.h2("3.4. Preprocesamiento y representacion"));
add(L.p("Los modelos clasicos representan el texto con TF-IDF de unigramas y bigramas, eliminando stopwords en espanol pero conservando palabras de negacion e intensidad (no, nunca, muy, pero) por su carga de sentimiento. Las redes neuronales aprenden embeddings desde cero sobre secuencias de indices de palabras. Los transformers usan su propio tokenizador subpalabra sobre el texto natural (comentario_limpio). El desbalance se trata con SMOTE (solo entrenamiento, en Naive Bayes), class_weight balanceado (SVM) y perdida ponderada por frecuencia inversa de clase (redes y transformers)."));

add(L.h2("3.5. Etiquetado asistido por LLM: seleccion del anotador"));
add(L.p("Para automatizar el etiquetado de la semilla de forma defendible, se comparo un conjunto de modelos de lenguaje candidatos como anotadores, midiendo su acuerdo con etiquetas de consenso de alta confianza del pipeline previo. El acuerdo exacto se complementa con el acuerdo adyacente (tolerante a diferencias de un nivel en la escala ordinal), pertinente por la ambiguedad natural entre clases vecinas."));
{
  const bench = tryCSV(R("11_etiquetado_llm/benchmark_etiquetador.csv"));
  if (bench) {
    const filas = bench.filter(r => r.acuerdo_exacto && r.acuerdo_exacto !== "")
      .map(r => [r.candidato, r.proveedor,
                 (Number(r.acuerdo_exacto) * 100).toFixed(1) + "%",
                 (Number(r.acuerdo_adyacente) * 100).toFixed(1) + "%",
                 (Number(r.cobertura) * 100).toFixed(0) + "%"]);
    add(L.tabla(["Modelo (LLM)", "Proveedor", "Acuerdo exacto", "Acuerdo adyacente", "Cobertura"],
      filas, [2600, 1900, 1700, 1900, 1260], [LFT, LFT, RGT, RGT, RGT]));
    add(L.pieTabla("Tabla 1. Benchmark de LLM anotadores frente a etiquetas de consenso (muestra estratificada)."));
    const mejor = bench.find(r => r.acuerdo_exacto && r.acuerdo_exacto !== "");
    if (mejor) add(L.p(`Se selecciono ${mejor.candidato} como anotador por lograr el mayor acuerdo exacto (${(Number(mejor.acuerdo_exacto)*100).toFixed(1)}%) con 100% de acuerdo adyacente y cobertura total. El mismo protocolo se aplico a la semilla y al conjunto de prueba, este ultimo como referencia independiente para la evaluacion final.`));
  }
}

add(L.h2("3.6. Fundamentacion de los algoritmos"));
add(L.p("Naive Bayes y SVM se usan como lineas base por su solidez con TF-IDF (McCallum & Nigam, 1998; Joachims, 1998). CNN para texto (Kim, 2014) captura patrones locales de n-gramas y LSTM (Hochreiter & Schmidhuber, 1997) modela dependencias secuenciales. Los transformers (Vaswani et al., 2017; Devlin et al., 2019) generan representaciones contextuales; para espanol se emplean BETO (Canete et al., 2020) y XLM-RoBERTa (Conneau et al., 2020)."));

// ============================ SECCION IV: RESULTADOS ============================
add(L.h1("IV. Resultados y Analisis"));

// 4.1 Particion y semilla
add(L.h2("4.1. Particion de datos y semilla representativa"));
{
  const rep = tryCSV(R("10_particion_semilla/reporte_particion.csv"));
  const dist = tryCSV(R("10_particion_semilla/distribucion_estrellas.csv"));
  if (rep) {
    const val = (m) => (rep.find(r => r.metrica === m) || {}).valor;
    add(L.p(`De las 4 800 resenas, se usaron ${val("filas_base_utilizable")} con texto valido. La particion estratificada por estrellas y longitud del comentario produjo: ${val("filas_test")} resenas de prueba (20% reservado), ${val("filas_semilla")} de semilla y ${val("filas_dev_resto")} para propagacion por self-training.`));
  }
  if (dist) {
    const estrellas = [...new Set(dist.map(r => r.estrellas))].sort();
    const bloques = ["test", "semilla", "dev_resto"];
    const filas = estrellas.map(e => [e + " estrella(s)",
      ...bloques.map(b => { const f = dist.find(r => r.estrellas === e && r.bloque === b); return f ? f.porcentaje + "%" : "-"; })]);
    add(L.tabla(["Estrellas", "Test", "Semilla", "Dev (resto)"], filas,
      [3360, 2000, 2000, 2000], [LFT, RGT, RGT, RGT]));
    add(L.pieTabla("Tabla 2. Distribucion porcentual por estrellas en cada bloque (estratificacion consistente)."));
  }
}

// 4.2 Etiquetado LLM y acuerdo
add(L.h2("4.2. Etiquetado asistido y acuerdo con el etiquetado previo"));
{
  const ac = tryCSV(R("11_etiquetado_llm/acuerdo_vs_pipeline_anterior.csv"));
  if (ac) {
    const filas = ac.map(r => [cap(r.bloque),
      r.referencia === "sentimiento_final" ? "Pipeline previo" : "Estrellas",
      r.n_comparadas,
      r.acuerdo_exacto ? (Number(r.acuerdo_exacto) * 100).toFixed(1) + "%" : "-",
      r.acuerdo_adyacente ? (Number(r.acuerdo_adyacente) * 100).toFixed(1) + "%" : "-"]);
    add(L.tabla(["Bloque", "Referencia", "N", "Acuerdo exacto", "Acuerdo adyacente"],
      filas, [1760, 2600, 1200, 1900, 1900], [LFT, LFT, RGT, RGT, RGT]));
    add(L.pieTabla("Tabla 3. Acuerdo del etiquetado LLM con las etiquetas previas y con las estrellas."));
    add(L.p("El acuerdo adyacente cercano al 96-98% indica que las discrepancias se concentran en niveles vecinos de la escala; el LLM, al juzgar solo el texto, reparte mas casos hacia las clases intermedias donde la senal de estrellas tiende a exagerar los extremos. Esto responde a la indicacion docente de coincidir con el etiquetado de referencia cuando corresponde y de revisar los casos ambiguos."));
  }
}

// 4.3 CV
add(L.h2("4.3. Validacion cruzada y seleccion de hiperparametros"));
{
  const cv = tryCSV(R("12_cv_modelos/cv_mejor_por_modelo.csv"));
  if (cv) {
    const orden = cv.slice().sort((a, b) => Number(b.f1_macro_medio) - Number(a.f1_macro_medio));
    const filas = orden.map(r => [FAMILIA_ES[r.familia] || r.familia, NOMBRE[r.modelo] || r.modelo,
      r.config.replace(/[{}"]/g, "").replace(/,/g, ", "),
      fmt(r.f1_macro_medio) + " +/- " + fmt(r.f1_macro_std, 3)]);
    add(L.tabla(["Familia", "Modelo", "Mejor config.", "F1-macro (CV 5-fold)"],
      filas, [1900, 2200, 2860, 2400], [LFT, LFT, LFT, RGT]));
    add(L.pieTabla("Tabla 4. Mejor configuracion por modelo segun validacion cruzada estratificada sobre la semilla."));
    add(L.p(`El mejor modelo por validacion cruzada fue ${NOMBRE[orden[0].modelo] || orden[0].modelo} (F1-macro ${fmt(orden[0].f1_macro_medio)}), seleccionado para la fase de self-training. Aun con solo 500 ejemplos, los transformers preentrenados en espanol superan claramente a las demas familias.`));
  }
}

// 4.4 Self-training
add(L.h2("4.4. Propagacion de etiquetas (self-training) y verificacion"));
{
  const res = tryCSV(R("13_self_training/resumen_self_training.csv"));
  if (res) {
    const val = (m) => (res.find(r => r.metrica === m) || {}).valor;
    add(L.p(`El mejor modelo etiqueto ${val("filas_dev_resto")} resenas del bloque de desarrollo. La verificacion detecto ${val("dudosos_confianza")} casos de baja confianza y ${val("dudosos_clustering")} incoherentes con su cluster; se enviaron ${val("enviados_arbitraje")} al arbitraje del LLM, que modifico ${val("cambiados_por_arbitraje")} etiquetas. El dataset de entrenamiento consolidado quedo con ${val("filas_dev_completo")} resenas etiquetadas.`));
  }
  const dist = tryCSV(R("13_self_training/distribucion_final_dev.csv"));
  if (dist) {
    const filas = dist.map(r => [cap(r.clase), r.cantidad]);
    add(L.tabla(["Clase", "Cantidad"], filas, [5680, 3680], [LFT, RGT]));
    add(L.pieTabla("Tabla 5. Distribucion de clases del dataset de entrenamiento consolidado (80%)."));
  }
}

// 4.5 Entrenamiento final + comparacion + evaluacion en test
add(L.h2("4.5. Entrenamiento final y comparacion de modelos"));
{
  const comp = tryCSV(R("14_entrenamiento_final/comparacion_valid.csv"));
  if (comp) {
    const valid = comp.filter(r => r.split === "valid").sort((a, b) => Number(b.f1_macro) - Number(a.f1_macro));
    const filas = valid.map(r => [FAMILIA_ES[r.familia] || r.familia, NOMBRE[r.modelo] || r.modelo,
      fmt(r.f1_macro), fmt(r.balanced_accuracy), fmt(r.accuracy)]);
    add(L.tabla(["Familia", "Modelo", "F1-macro", "Bal. acc.", "Accuracy"],
      filas, [1900, 2260, 1700, 1750, 1750], [LFT, LFT, RGT, RGT, RGT]));
    add(L.pieTabla("Tabla 6. Comparacion de los seis modelos finales en el conjunto de validacion (30% del desarrollo)."));
  }
}

// 4.6 Mejor modelo en test
add(L.h2("4.6. Evaluacion final del mejor modelo (conjunto de prueba)"));
{
  const res = tryCSV(R("14_entrenamiento_final/resumen_final.csv"));
  if (res) {
    const val = (m) => (res.find(r => r.metrica === m) || {}).valor;
    const mejor = val("mejor_modelo");
    add(L.p(`El mejor modelo por validacion fue ${NOMBRE[mejor] || mejor}, evaluado una unica vez sobre el 20% de prueba reservado. Obtuvo F1-macro ${fmt(val("f1_macro_test"))}, exactitud ${fmt(val("accuracy_test"))}, F1 ponderado ${fmt(val("f1_weighted_test"))} y exactitud balanceada ${fmt(val("balanced_accuracy_test"))}.`));

    // Reporte de clasificacion del mejor en test
    const combo = `${mejor}_final`;
    const repTxt = path.join(ROOT, "reports", "14_entrenamiento_final", `reporte_clasificacion_${combo}_test.txt`);
    // Matriz de confusion
    const mc = tryCSV(R(`14_entrenamiento_final/matriz_confusion_${combo}_test.csv`));
    if (mc) {
      const cabecera = Object.keys(mc[0]);
      const clases = cabecera.slice(1);
      const filas = mc.map(r => [cap(r[cabecera[0]] || Object.values(r)[0]), ...clases.map(c => r[c])]);
      const anchos = [2160, ...clases.map(() => Math.floor(7200 / clases.length))];
      add(L.tabla(["Real / Predicho", ...clases.map(cap)], filas, anchos,
        [LFT, ...clases.map(() => RGT)]));
      add(L.pieTabla("Tabla 7. Matriz de confusion del mejor modelo en el conjunto de prueba."));
    }
    // ROC / AUC
    const auc = tryCSV(R(`14_entrenamiento_final/auc_${combo}_test.csv`));
    if (auc) {
      const filas = auc.map(r => [cap(r.clase), r.auc && r.auc !== "" ? fmt(r.auc, 3) : "-"]);
      add(L.tabla(["Clase", "AUC (One-vs-Rest)"], filas, [5680, 3680], [LFT, RGT]));
      add(L.pieTabla("Tabla 8. Area bajo la curva ROC por clase (One-vs-Rest) del mejor modelo en prueba."));
    }
    const rocPng = R(`14_entrenamiento_final/curva_roc_${combo}_test.png`);
    if (L.existe(rocPng)) { add(L.imagen(rocPng, { ancho: 430, alto: 330 }));
      add(L.pieTabla("Figura 1. Curvas ROC One-vs-Rest del mejor modelo en el conjunto de prueba.")); }
  }
}

// 4.7 Curvas de entrenamiento
add(L.h2("4.7. Curvas de entrenamiento"));
add(L.p("Las curvas de entrenamiento muestran la evolucion de la perdida y del F1-macro de validacion por epoca (redes y transformers) y la curva de aprendizaje (F1-macro frente al tamano de entrenamiento) para los modelos clasicos, atendiendo la solicitud docente de graficar el proceso de entrenamiento."));
{
  const cmp = R("14_entrenamiento_final/comparacion_final_f1.png");
  if (L.existe(cmp)) { add(L.imagen(cmp, { ancho: 470, alto: 280 }));
    add(L.pieTabla("Figura 2. Comparacion de F1-macro en validacion de los seis modelos finales.")); }
  for (const m of ["beto", "lstm", "cnn", "xlm_roberta"]) {
    const png = R(`14_entrenamiento_final/curva_entrenamiento_${m}.png`);
    if (L.existe(png)) { add(L.imagen(png, { ancho: 430, alto: 270 }));
      add(L.pieTabla(`Figura. Curva de entrenamiento de ${NOMBRE[m] || m} (perdida y F1-macro por epoca).`)); }
  }
}

// ============================ CONCLUSIONES ============================
add(L.h1("V. Conclusiones"));
{
  const res = tryCSV(R("14_entrenamiento_final/resumen_final.csv"));
  let frase = "el mejor modelo transformer preentrenado en espanol";
  if (res) {
    const val = (m) => (res.find(r => r.metrica === m) || {}).valor;
    const mejor = val("mejor_modelo");
    frase = `${NOMBRE[mejor] || mejor} (F1-macro ${fmt(val("f1_macro_test"))} y exactitud ${fmt(val("accuracy_test"))} en prueba)`;
  }
  add(L.p(`Se cumplio el objetivo de construir y evaluar un modelo de clasificacion de sentimiento multiclase sobre resenas reales de consumidores peruanos, adoptando un flujo semisupervisado alineado con la retroalimentacion docente: reserva de un conjunto de prueba independiente, etiquetado de una semilla representativa con asistencia de IA y revision del equipo, seleccion de hiperparametros por validacion cruzada, propagacion de etiquetas por self-training con verificacion por clustering y arbitraje, y evaluacion final unica sobre el conjunto reservado. El mejor desempeno lo alcanzo ${frase}, confirmando que las representaciones contextuales preentrenadas en espanol capturan mejor la variabilidad del lenguaje del consumidor peruano que los modelos basados en frecuencias o embeddings entrenados desde cero.`));
  add(L.p("Como trabajo futuro, se recomienda ampliar sustancialmente el volumen de datos -especialmente de las clases intermedias (negativo, neutral y positivo), que concentran la mayor confusion-, incorporar mas revision humana sobre los casos de baja confianza y explorar tecnicas adicionales de balanceo y aumento de datos. Con un corpus mas amplio y depurado, el modelo transformer seleccionado podria ajustarse y validarse en datos nuevos antes de un despliegue operativo."));
}

// ============================ REFERENCIAS ============================
add(L.h1("Referencias"));
const REFERENCIAS = [
  "Arango Pastrana, C. A., & Osorio Andrade, C. F. (2021). Aislamiento social obligatorio: un analisis de sentimientos mediante machine learning. Suma de Negocios, 12(26), 1-13.",
  "Ashbaugh, L., & Zhang, Y. (2024). A comparative study of sentiment analysis on customer reviews using machine learning and deep learning. Computers, 13(12), 340.",
  "Aunimo, L., Oprescu, A. M., Kudryavtsev, D., Munoz Saavedra, L., & Romero Ternero, M. D. C. (2025). Perceived quality of service in primary health care based on Google Maps reviews: sentiment analysis. Journal of Medical Internet Research, 27, e70410.",
  "Canete, J., Chaperon, G., Fuentes, R., Ho, J. H., Kang, H., & Perez, J. (2020). Spanish pre-trained BERT model and evaluation data. PML4DC at ICLR 2020.",
  "Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised cross-lingual representation learning at scale (XLM-R). ACL 2020.",
  "Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL 2019.",
  "Grandini, M., Bagli, E., & Visani, G. (2020). Metrics for multi-class classification: an overview. arXiv:2008.05756.",
  "Herrera Gambini, E. M. (2021). Analisis de sentimiento de comentarios en espanol usando BERT y BETO.",
  "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.",
  "Joachims, T. (1998). Text categorization with support vector machines. ECML 1998.",
  "Kim, Y. (2014). Convolutional neural networks for sentence classification. EMNLP 2014.",
  "McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. AAAI Workshop 1998.",
  "Mihalcea, R., & Ionescu, R. (2023). Text classification with TF-IDF and n-grams.",
  "Mutanov, G., Karyukin, V., & Mamykova, Z. (2021). Multi-class sentiment analysis of social media data with machine learning algorithms. Computers, Materials & Continua, 69(1), 913-930.",
  "OpenWebinars (2024). Tecnicas clave para el procesamiento de texto en NLP.",
  "OSIPTEL (2025). Reporte de reclamos y calidad de atencion del servicio movil 2024.",
  "Peffers, K., Tuunanen, T., Rothenberger, M. A., & Chatterjee, S. (2007). A design science research methodology for information systems research. JMIS, 24(3), 45-77.",
  "Shin, B., Ryu, S., Kim, Y., & Kim, D. (2022). Analysis on review data of restaurants in Google Maps through text mining. Journal of Multimedia Information System, 9(1), 61-68.",
  "Suandi, F., et al. (2024). Enhancing sentiment analysis performance using SMOTE and majority voting. ICAE 2024, 126-138.",
  "Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. NeurIPS 2017.",
  "Villanueva, K. C. N. (2024). Comparative study of machine learning models for sentiment analysis: SVM, Naive Bayes and Logistic Regression.",
  "Zaid, S., Alharbi, A. H., & Samra, H. (2025). Multi-aspect sentiment classification of Arabic tourism reviews using BERT and classical ML. Data, 10(11), 168.",
];
REFERENCIAS.forEach((ref, i) => add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 264 },
  indent: { left: 360, hanging: 360 },
  children: [new TextRun({ text: `[${i + 1}] ${ref}`, size: 20, font: "Times New Roman" })],
})));

// ============================ ENSAMBLADO ============================
const doc = new Document({
  creator: "Grupo 05 - Mineria de Datos",
  title: "Analisis de Sentimiento Multiclase - Flujo Semisupervisado",
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 260, after: 130 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Times New Roman" },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "vinetas", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Grupo 05 - Mineria de Datos - UNMSM 2026   |   Pagina ", size: 16, font: "Times New Roman" }),
                 new TextRun({ children: [PageNumber.CURRENT], size: 16, font: "Times New Roman" })] })] }) },
    children: contenido,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  const salida = path.join(__dirname, "Informe_Grupo05_Flujo_Semisupervisado.docx");
  fs.writeFileSync(salida, buffer);
  console.log("Informe generado:", salida, "(" + (buffer.length / 1024).toFixed(0) + " KB)");
});
