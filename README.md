# Analisis De Sentimiento Multiclase Sobre Resenas De Empresas Peruanas

Proyecto para construir, limpiar, etiquetar y preparar un dataset de resenas de consumidores peruanos para clasificacion de sentimiento multiclase.

El flujo actual no depende solo de estrellas de Google. Usa una estrategia por etapas:

```text
scraping
limpieza
autoetiquetado con pysentimiento
revision asistida por IA de casos ambiguos
integracion conservadora de etiquetas
etiquetado automatico por reglas (sin IA, determinista)
split estratificado
modelado y evaluacion (tres familias de modelos)
comparacion unificada
```

El modelado compara **seis modelos en tres familias**, todos sobre el mismo split y
evaluados con el mismo criterio:

```text
clasicos      (fase 07) -> SVM, Naive Bayes            (TF-IDF)
deep learning (fase 08) -> CNN, LSTM                   (embeddings entrenables, PyTorch)
transformers  (fase 09) -> BETO, XLM-RoBERTa           (fine-tuning, HuggingFace)
comparacion   (fase 10) -> tabla y grafico unificados de las tres familias
```

## Metodologia

El proyecto combina dos metodologias: el marco general del proceso de mineria de datos
y la metodologia de etiquetado de los datos.

### Marco General: CRISP-DM

El pipeline sigue la estructura de CRISP-DM (Cross-Industry Standard Process for Data Mining):

```text
Comprension del negocio  -> objetivo: clasificar sentimiento multiclase de resenas peruanas
Comprension de los datos -> scraping y auditoria (etapas 01 y 02-auditoria)
Preparacion de los datos -> limpieza, autoetiquetado, revision IA, integracion, reglas, split (02-06, 05b)
Modelado                 -> tres familias: clasicos (07), deep learning (08) y transformers (09)
Evaluacion               -> F1-macro, exactitud balanceada, F1 por clase, matriz de confusion + comparacion (10)
Despliegue               -> script de prediccion (predecir.py) sobre el mejor modelo clasico
```

El ciclo de preparacion -> modelado -> evaluacion ya esta cerrado y abarca las tres familias de
modelos. La preparacion de datos sigue siendo la mayor parte del esfuerzo.

### Metodologia De Etiquetado: Supervision Debil Con Consenso

Las etiquetas no se asignan a mano una por una ni confiando ciegamente en una sola fuente.
Se usa un esquema de supervision debil (weak supervision) que combina varias tecnicas:

```text
Supervision distante / etiquetas debiles  -> las estrellas de Google como etiqueta inicial
Pre-etiquetado asistido por modelo         -> pysentimiento aporta una segunda senal
Etiquetado por consenso                    -> se confia solo cuando estrellas y modelo coinciden
Revision asistida por IA (human-in-loop)   -> los casos ambiguos se revisan de forma focalizada
Funciones de etiquetado por reglas         -> reglas deterministas validadas, estilo Snorkel (fase 05b)
Abstencion conservadora                    -> los casos contradictorios se dejan sin etiqueta
```

Criterio rector: priorizar la pureza de las etiquetas sobre la cobertura. Ver el detalle de las
fuentes de etiqueta en la seccion "Criterio De Etiquetado".

### Validacion Y Particion

- Validacion de etiquetas: se mide el acuerdo de las reglas frente a las etiquetas humanas ya
  existentes antes de confiar en ellas (actualmente 93% de acuerdo, ver fase 05b).
- Particion estratificada train/valid/test que preserva la distribucion de clases,
  metodologia estandar para clasificacion con desbalance. Esquema 80% desarrollo / 20% prueba,
  subdividiendo el desarrollo en 70% entrenamiento / 30% validacion (resultado: 56/24/20 sobre
  el total).

## Instalacion

```bash
pip install -r requirements.txt
```

## Pruebas

La logica pura de etiquetado y normalizacion esta cubierta con tests. Para correrlos:

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

Codigo compartido reutilizable en `scripts/_comun/`:

```text
_comun/etiquetas.py    -> MAPEO_ESTRELLAS, MAPEO_MODELO y helpers de estrellas
_comun/texto.py        -> normalizacion de texto (normalizar_texto, texto_para_modelo, ...)
_comun/dataframes.py   -> validar_columnas y mapeo de etiquetas IA al dataset
```

## Ejecucion Con Un Solo Comando

En lugar de correr las etapas una por una, el orquestador `run_pipeline.py` encadena todo
el procesamiento (limpieza -> autoetiquetado -> integracion -> reglas -> split) con un solo
comando. Verifica las entradas de cada etapa y salta con un mensaje claro las que no puedan
correr (falta de archivo o de paquete), mostrando un resumen final.

```bash
python run_pipeline.py                 # corre el procesamiento 02 -> 06
python run_pipeline.py --listar        # muestra las etapas y si pueden correr
python run_pipeline.py --dry-run       # muestra que haria, sin ejecutar nada
python run_pipeline.py --instalar-deps # instala requirements.txt antes de correr
python run_pipeline.py --con-scraping  # incluye la etapa 01 (necesita Chrome abierto)
python run_pipeline.py --desde 05b_reglas           # corre desde una etapa
python run_pipeline.py --solo 05b_reglas 06_split   # corre solo esas etapas
```

El scraping (01) esta excluido por defecto porque necesita que abras Chrome con depuracion
remota a mano (ver mas abajo). El autoetiquetado (03) requiere `pysentimiento` instalado y el
dataset crudo ya scrapeado; si faltan, esas etapas se saltan y el pipeline continua con las
demas. Las secciones siguientes documentan cada etapa por separado para ejecucion manual.

## Estructura Del Proyecto

```text
data/
  raw/
    empresas.csv
    dataset_consumidores_peru.csv
  processed/
    dataset_consumidores_peru_limpio.csv
    dataset_consumidores_peru_etiquetado.csv
    dataset_consumidores_peru_etiquetado_final.csv
  splits/

models/
  mejor_modelo.joblib            (mejor clasico, fase 07)
  mejor_modelo_dl.pt             (mejor red, fase 08)
  mejor_modelo_transformer/      (mejor transformer, fase 09)

reports/
  01_limpieza/
  02_autoetiquetado/
  03_revision_ia/
  03b_reglas/
  04_integracion/
  05_split/
  06_modelado/        (clasicos: SVM, Naive Bayes)
  07_dl/              (deep learning: CNN, LSTM)
  08_transformers/    (transformers: BETO, XLM-RoBERTa)
  09_comparacion/     (comparacion unificada de las 3 familias)

scripts/
  _comun/             (codigo compartido: texto, datos, evaluacion, etiquetas)
  01_scraping/
  02_limpieza/
  03_autoetiquetado/
  04_revision_ia/
  05_integracion/
  06_split/
  07_modelado/        (SVM, Naive Bayes)
  08_dl/              (CNN, LSTM)
  09_transformers/    (BETO, XLM-RoBERTa)
  10_comparacion/     (comparacion unificada)
```

## Estado Actual Del Dataset

Archivo final actual:

```text
data/processed/dataset_consumidores_peru_etiquetado_final.csv
```

Distribucion actual de `sentimiento_final`:

```text
muy positivo    1407
muy negativo    1054
positivo         647
neutral          639
negativo         433
```

Total con etiqueta final consolidada:

```text
4180
```

Casos sin etiqueta final confiable:

```text
620
```

Estas cifras incluyen 390 etiquetas asignadas por el etiquetado automatico por reglas
(fase 05b), que recupera neutrales y negativos de alta pureza sin usar IA. Ver mas abajo.

## Pipeline Completo

### 01. Scraping

Antes de ejecutar el scraper, abre Chrome con debugging remoto:

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome-debug"
```

Si Chrome esta instalado en `Program Files (x86)`, usa:

```powershell
& "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome-debug"
```

Ejecutar scraping:

```bash
python scripts/01_scraping/scriptscraping.py --max-reviews 150
```

Entrada:

```text
data/raw/empresas.csv
```

Salida:

```text
data/raw/dataset_consumidores_peru.csv
```

El archivo `empresas.csv` debe tener estas columnas:

```text
nombre,sede,rubro,url
```

El scraper incluye medidas anti-bloqueo para corridas largas: delays con jitter (intervalos
aleatorios en vez de constantes), pausas humanas entre sedes y deteccion de la pagina de
verificacion de Google con backoff exponencial (se frena en vez de insistir). Para ir mas lento
y seguro, ajusta las pausas:

```bash
python scripts/01_scraping/scriptscraping.py --max-reviews 150 --pausa-sede-min 8 --pausa-sede-max 20
```

Banderas anti-bloqueo: `--pausa-sede-min`, `--pausa-sede-max` (segundos entre sedes) y
`--max-bloqueos` (detiene el scraping tras N bloqueos consecutivos; el CSV acumulado se conserva
y el scraper salta las sedes ya completas al reanudar).

### 02. Limpieza Y Auditoria

```bash
python scripts/02_limpieza/limpiar_dataset.py
python scripts/02_limpieza/auditar_dataset.py
```

Entrada:

```text
data/raw/dataset_consumidores_peru.csv
```

Salidas:

```text
data/processed/dataset_consumidores_peru_limpio.csv
reports/01_limpieza/reporte_limpieza_dataset.csv
reports/01_limpieza/distribucion_sentimiento.csv
reports/01_limpieza/distribucion_estrellas.csv
```

La limpieza conserva el texto original y agrega columnas de auditoria como:

```text
comentario_limpio
texto_modelo
comentario_corto
sin_contenido_alfabetico
sentimiento_consistente
requiere_revision
motivo_revision
```

### 03. Autoetiquetado Con Modelo En Espanol

```bash
python scripts/03_autoetiquetado/autoetiquetar_sentimiento.py
```

Entrada:

```text
data/processed/dataset_consumidores_peru_limpio.csv
```

Salidas:

```text
data/processed/dataset_consumidores_peru_etiquetado.csv
reports/02_autoetiquetado/reporte_autoetiquetado.csv
reports/02_autoetiquetado/distribucion_sentimiento_final.csv
reports/02_autoetiquetado/distribucion_sentimiento_provisional.csv
```

Esta fase usa `pysentimiento` sobre `comentario_limpio` y compara el resultado con las estrellas.

Columnas principales agregadas:

```text
sentimiento_modelo
polaridad_modelo
confianza_modelo
prob_neg
prob_neu
prob_pos
sentimiento_final
sentimiento_final_provisional
confianza_etiqueta
requiere_revision_etiqueta
motivo_revision_etiqueta
```

### 04. Revision Asistida Por IA

Preparar los casos ambiguos prioritarios:

```bash
python scripts/04_revision_ia/preparar_revision_etiquetas.py --max-filas 500
python scripts/04_revision_ia/preparar_revision_para_ia.py
```

Archivos para IA:

```text
reports/03_revision_ia/revision_prioritaria_para_ia.csv
reports/03_revision_ia/prompt_etiquetado_ia.md
```

La IA debe devolver:

```text
reports/03_revision_ia/revision_prioritaria_etiquetas_ia.csv
```

Preparar segundo lote enfocado en posibles neutrales:

```bash
python scripts/04_revision_ia/preparar_revision_neutral_para_ia.py
```

Archivos para IA:

```text
reports/03_revision_ia/revision_neutral_para_ia.csv
reports/03_revision_ia/prompt_etiquetado_ia_neutral.md
```

La IA debe devolver:

```text
reports/03_revision_ia/revision_neutral_etiquetas_ia.csv
```

Revision manual opcional:

```bash
python scripts/04_revision_ia/revisar_etiquetas_manual.py --limite 50
```

### 05. Integracion De Etiquetas IA

Integrar etiquetas IA de clases minoritarias:

```bash
python scripts/05_integracion/integrar_etiquetas_ia.py
```

Integrar neutrales del segundo lote:

```bash
python scripts/05_integracion/integrar_etiquetas_ia_neutral.py
```

Salidas:

```text
data/processed/dataset_consumidores_peru_etiquetado_final.csv
reports/04_integracion/reporte_integracion_ia.csv
reports/04_integracion/reporte_integracion_ia_neutral.csv
reports/04_integracion/distribucion_sentimiento_final_integrado.csv
```

Regla usada en integracion:

```text
usar solo etiquetas IA con usar_etiqueta_ia=si
usar solo confianza alta o media
integrar de forma focalizada clases minoritarias
```

### 05b. Etiquetado Automatico Por Reglas

Esta fase rellena casos que quedaron sin etiqueta usando solo columnas ya calculadas
(`estrellas`, `sentimiento_modelo`, `prob_neu`, `prob_pos`). No ejecuta ningun modelo de IA:
es determinista, reproducible y solo depende de `pandas`.

```bash
python scripts/04_revision_ia/etiquetar_por_reglas.py --dry-run
python scripts/04_revision_ia/etiquetar_por_reglas.py
```

Entrada:

```text
data/processed/dataset_consumidores_peru_etiquetado_final.csv
```

Salidas:

```text
data/processed/dataset_consumidores_peru_etiquetado_final.csv   (actualizado en sitio)
reports/03b_reglas/reporte_etiquetado_reglas.csv
reports/03b_reglas/distribucion_final_reglas.csv
reports/03b_reglas/validacion_reglas.csv
```

Reglas conservadoras (solo sobre filas con `sentimiento_final` vacio):

```text
R1: 4/5 estrellas + modelo NEU con prob_neu >= 0.60  -> neutral
R2: 1 estrella  + modelo NEU + prob_pos < 0.25       -> muy negativo
    2 estrellas + modelo NEU + prob_pos < 0.25       -> negativo
```

R1 se valido empiricamente: cuando hay muchas estrellas pero el modelo dice NEU con confianza,
la resena es genuinamente neutral (cliente generoso con la estrella, texto con peros), no positiva.
El script compara lo que la regla predeciria contra las etiquetas IA ya existentes y reporta el
porcentaje de acuerdo (actualmente 93%). Las contradicciones duras (estrellas altas + NEG,
estrellas bajas + POS, y 3 estrellas residual) se dejan sin etiqueta a proposito.

Filas marcadas con `sentimiento_final_origen = regla_voto_ponderado` para trazabilidad. Banderas:
`--umbral-neu`, `--umbral-pos`, `--dry-run`.

### 06. Split Estratificado

```bash
python scripts/06_split/preparar_split_dataset.py
```

Entrada:

```text
data/processed/dataset_consumidores_peru_etiquetado_final.csv
```

Salidas:

```text
data/splits/train.csv
data/splits/valid.csv
data/splits/test.csv
reports/05_split/reporte_split_dataset.csv
reports/05_split/distribucion_split_train.csv
reports/05_split/distribucion_split_valid.csv
reports/05_split/distribucion_split_test.csv
```

Tamanos actuales (split 80/20 + 70/30, `random_state=42`):

```text
train    2338
valid    1003
test      836
```

### 07. Modelos Clasicos: SVM Y Naive Bayes

```bash
python scripts/07_modelado/entrenar_evaluar.py
```

Entrada:

```text
data/splits/train.csv
data/splits/valid.csv
data/splits/test.csv
```

Salidas:

```text
reports/06_modelado/comparacion_modelos.csv
reports/06_modelado/f1_por_clase.csv
reports/06_modelado/matriz_confusion_<modelo>_<estrategia>_test.csv / .png
reports/06_modelado/reporte_clasificacion_<modelo>_<estrategia>_test.txt
models/mejor_modelo.joblib
```

El clasificador usa TF-IDF (1-2 gramas, con eliminacion de stopwords en espanol pero conservando
las palabras de negacion e intensidad) sobre `texto_modelo`. Esta fase cubre los **dos modelos
clasicos** del enunciado, cruzados con tres estrategias frente al desbalance:

```text
modelos:     SVM (LinearSVC), Naive Bayes (MultinomialNB)
estrategias: base | balanced (class_weight) | smote (solo en train)
```

(Naive Bayes no soporta class_weight, asi que su combinacion con `balanced` se omite.)

Metricas: F1-Macro (principal), exactitud balanceada, F1 por clase, matriz de confusion y accuracy.
La mejor combinacion se elige por F1-Macro en validacion y se guarda con joblib.

Resultados actuales (F1-Macro en test, ordenado):

```text
modelo       estrategia   valid    test
naive_bayes  smote        0.5009   0.4865   <- mejor por F1-macro en validacion (se guarda)
svm          smote        0.4852   0.4799
svm          balanced     0.4953   0.4785
svm          base         0.4898   0.4587
naive_bayes  base         0.2970   0.2925
```

Hallazgos: (1) el balanceo es decisivo: el Naive Bayes base tiene buena accuracy pero F1-Macro
pobre (0.29); (2) **SMOTE rescata a Naive Bayes** (0.29 -> 0.49 en F1-Macro), coherente con la
literatura citada en el documento, y lo convierte en el mejor clasico; (3) SVM es estable y
competitivo en validacion; (4) la mayor confusion ocurre entre clases adyacentes (positivo vs
muy positivo), patron ordinal esperable. Quitar stopwords baja levemente el F1 porque parte de
la senal vive en palabras funcionales; se mantiene por alineacion metodologica y existe la
bandera `--sin-stopwords`.

Para reentrenar solo algunas combinaciones:

```bash
python scripts/07_modelado/entrenar_evaluar.py --algoritmos svm --estrategias balanced smote
```

### 08. Modelos De Deep Learning: CNN Y LSTM

```bash
python scripts/08_dl/entrenar_dl.py
```

Entrada: los mismos splits que la fase 07. Salidas:

```text
reports/07_dl/comparacion_dl.csv
reports/07_dl/f1_por_clase_dl.csv
reports/07_dl/matriz_confusion_<modelo>_<estrategia>_test.csv / .png
reports/07_dl/reporte_clasificacion_<modelo>_<estrategia>_test.txt
models/mejor_modelo_dl.pt
```

Dos arquitecturas en PyTorch, con embeddings **entrenados desde cero** (sin vectores
preentrenados, coherente con un dataset pequeno y especifico) sobre `texto_modelo`:

```text
CNN  -> TextCNN: embeddings + Conv1d con kernels 3/4/5 + max-pooling + densa
LSTM -> BiLSTM: embeddings + LSTM bidireccional + mean-pooling enmascarado + densa
estrategias: base | class_weight (perdida ponderada por frecuencia inversa)
```

No se usa SMOTE: sobre secuencias de longitud variable no aplica de forma natural; el
equivalente estandar en deep learning es ponderar la perdida. El entrenamiento usa
early stopping por F1-macro en validacion y corre en GPU (CUDA) si esta disponible.

Resultados (F1-Macro en test, ordenado):

```text
modelo  estrategia    valid    test
cnn     class_weight  0.4831   0.5099
cnn     base          0.4684   0.4649
lstm    base          0.4938   0.4560   <- mejor por F1-macro en validacion (se guarda)
lstm    class_weight  0.4791   0.4535
```

Las redes quedan a la par de los clasicos: con ~3300 ejemplos de entrenamiento no hay datos
suficientes para que el deep learning desde cero supere claramente a TF-IDF.

### 09. Modelos Transformer: BETO Y XLM-RoBERTa

```bash
python scripts/09_transformers/entrenar_transformers.py
```

Entrada: los mismos splits, pero usando `comentario_limpio` (texto natural, con mayusculas
y signos) porque los transformers traen su propio tokenizador subpalabra. Salidas:

```text
reports/08_transformers/comparacion_transformers.csv
reports/08_transformers/f1_por_clase_transformers.csv
reports/08_transformers/matriz_confusion_<modelo>_<estrategia>_test.csv / .png
reports/08_transformers/reporte_clasificacion_<modelo>_<estrategia>_test.txt
models/mejor_modelo_transformer/
```

Fine-tuning de dos transformers preentrenados:

```text
BETO        -> dccuchile/bert-base-spanish-wwm-cased  (BERT en espanol, U. de Chile)
XLM-RoBERTa -> xlm-roberta-base                        (RoBERTa multilingue)
estrategia  -> class_weight (perdida ponderada); --estrategias base class_weight para ambas
```

Usa GPU con precision mixta (AMP), AdamW con warmup lineal y early stopping por F1-macro.
La primera corrida descarga los modelos de HuggingFace (~0.4 GB BETO, ~1.1 GB XLM-R).

Resultados (F1-Macro):

```text
modelo       estrategia    valid    test
beto         class_weight  0.6089   0.6258   <- mejor transformer en validacion y test (se guarda)
xlm_roberta  class_weight  0.6033   0.6039
```

Ambos transformers superan por amplio margen (entre 0.12 y 0.14 de F1-Macro) a clasicos y redes,
justo lo que anticipa la literatura citada en el documento del proyecto (BETO/BERT en espanol).
Con este split BETO lidera tanto en validacion como en prueba; la mejor combinacion se elige por
F1-macro en validacion (BETO) y se guarda completa.

### 10. Comparacion Unificada De Las Tres Familias

```bash
python scripts/10_comparacion/comparar_todos.py
```

Reune las comparaciones de las fases 07, 08 y 09 (todas escritas con el mismo esquema de
columnas, definido en `scripts/_comun/evaluacion.py`) en una sola tabla y un grafico. Salidas:

```text
reports/09_comparacion/comparacion_global.csv        (todas las filas)
reports/09_comparacion/comparacion_global_test.csv   (solo prueba, ordenado)
reports/09_comparacion/comparacion_global_f1_macro.png
```

Mejor combinacion por familia (F1-Macro en test):

```text
familia        mejor modelo (estrategia)    F1-macro test
transformer    beto (class_weight)          0.6258    <- mejor modelo global
deep_learning  cnn (class_weight)           0.5099
clasico        naive_bayes (smote)          0.4865
```

Conclusion del avance: el ranking es claro y estable -> **transformers > deep learning ~= clasicos**.
Con ~3000 ejemplos, las redes desde cero no superan a TF-IDF, pero el conocimiento preentrenado de
BETO y XLM-RoBERTa marca la diferencia. La mayor confusion en todas las familias ocurre entre clases
adyacentes (positivo vs muy positivo), patron ordinal esperable en una escala de 5 niveles.

## Criterio De Etiquetado

Las estrellas de Google se tratan como etiquetas debiles iniciales:

```text
1 estrella  -> muy negativo
2 estrellas -> negativo
3 estrellas -> neutral
4 estrellas -> positivo
5 estrellas -> muy positivo
```

La etiqueta final `sentimiento_final` se consolida mediante:

```text
1. coincidencia entre estrellas y modelo de sentimiento
2. revision asistida por IA focalizada en clases minoritarias
3. revision humana opcional de control
```

Los registros sin etiqueta confiable se mantienen en el dataset, pero no deben usarse para entrenamiento principal.

## Documentacion Extendida

El archivo `PIPELINE.md` mantiene el mismo flujo en formato resumido para consulta rapida.
