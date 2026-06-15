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
entrenamiento y evaluacion
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
Modelado                 -> TF-IDF + Regresion Logistica con 3 estrategias de balanceo (fase 07)
Evaluacion               -> F1-macro, F1 por clase, matriz de confusion, accuracy (fase 07)
Despliegue               -> no contemplado aun
```

El ciclo de preparacion -> modelado -> evaluacion ya esta cerrado. La preparacion de datos sigue
siendo la mayor parte del esfuerzo; el modelado actual es una linea base solida sobre la cual mejorar.

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
- Particion estratificada train/valid/test (70/15/15) que preserva la distribucion de clases,
  metodologia estandar para clasificacion con desbalance.

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

reports/
  01_limpieza/
  02_autoetiquetado/
  03_revision_ia/
  04_integracion/
  05_split/

scripts/
  01_scraping/
  02_limpieza/
  03_autoetiquetado/
  04_revision_ia/
  05_integracion/
  06_split/
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

Tamanos actuales:

```text
train    2923
valid     627
test      627
```

### 07. Entrenamiento Y Evaluacion

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
reports/06_modelado/f1_por_clase_<estrategia>_<split>.csv
reports/06_modelado/matriz_confusion_<estrategia>_<split>.csv / .png
reports/06_modelado/reporte_clasificacion_<estrategia>_<split>.txt
models/modelo_<mejor_estrategia>.joblib
```

El clasificador es TF-IDF (1-2 gramas) + Regresion Logistica sobre `texto_modelo`. Se comparan
tres estrategias frente al desbalance de clases:

```text
base       -> sin balanceo
balanced   -> LogisticRegression(class_weight='balanced')
smote       -> sobremuestreo SMOTE solo en train
```

Metrica principal: F1-Macro (justa con clases minoritarias). Secundarias: F1 por clase,
matriz de confusion y accuracy. La mejor estrategia se elige por F1-Macro en validacion y se
guarda el modelo entrenado (vectorizador + clasificador) con joblib.

Resultados actuales (F1-Macro):

```text
estrategia   valid    test
base         0.3834   0.4040
balanced     0.5048   0.5665   <- mejor
smote        0.4789   0.5517
```

El balanceo es decisivo: el modelo base tiene buena accuracy pero F1-Macro pobre (predice bien las
clases mayoritarias e ignora las minoritarias). `class_weight='balanced'` sube el F1-Macro de 0.40
a 0.57 en test. F1 por clase de la mejor estrategia (test): muy negativo 0.83, muy positivo 0.69,
neutral 0.49, negativo 0.48, positivo 0.35. La mayor confusion ocurre entre clases adyacentes
(positivo vs muy positivo), un patron ordinal esperable.

Para reentrenar solo una estrategia:

```bash
python scripts/07_modelado/entrenar_evaluar.py --estrategias balanced
```

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
