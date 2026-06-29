# Pipeline Del Proyecto

Este documento resume el orden de ejecucion del proyecto y los archivos principales de cada fase.

## 00. Comando Unico (orquestador)

Para correr todo el procesamiento de una sola vez, sin ejecutar cada etapa a mano:

```bash
python run_pipeline.py            # procesamiento 02 -> 06
python run_pipeline.py --listar   # estado de cada etapa
python run_pipeline.py --dry-run  # simula sin ejecutar
```

Banderas: `--con-scraping`, `--instalar-deps`, `--desde`, `--hasta`, `--solo`,
`--continuar-en-error`. El orquestador salta las etapas cuya entrada o paquete falte y
muestra un resumen final con OK / SALTADA / FALLO por etapa.

## 01. Scraping

Comando:

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

## 02. Limpieza Y Auditoria

Comandos:

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

## 03. Autoetiquetado

Comando:

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

## 04. Revision Asistida Por IA

Preparar casos ambiguos prioritarios:

```bash
python scripts/04_revision_ia/preparar_revision_etiquetas.py --max-filas 500
python scripts/04_revision_ia/preparar_revision_para_ia.py
```

Archivos para IA:

```text
reports/03_revision_ia/revision_prioritaria_para_ia.csv
reports/03_revision_ia/prompt_etiquetado_ia.md
```

Respuesta esperada de IA:

```text
reports/03_revision_ia/revision_prioritaria_etiquetas_ia.csv
```

Preparar segundo lote enfocado en neutral:

```bash
python scripts/04_revision_ia/preparar_revision_neutral_para_ia.py
```

Archivos para IA:

```text
reports/03_revision_ia/revision_neutral_para_ia.csv
reports/03_revision_ia/prompt_etiquetado_ia_neutral.md
```

Respuesta esperada de IA:

```text
reports/03_revision_ia/revision_neutral_etiquetas_ia.csv
```

Revision manual opcional:

```bash
python scripts/04_revision_ia/revisar_etiquetas_manual.py --limite 50
```

## 05. Integracion De Etiquetas IA

Integrar clases minoritarias del primer lote:

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

Estado actual de clases finales:

```text
muy positivo    1407
muy negativo    1054
positivo         647
neutral          639
negativo         433
```

## 05b. Etiquetado Automatico Por Reglas

Rellena filas sin etiqueta usando solo columnas ya calculadas (sin IA, determinista, solo pandas).

Comandos:

```bash
python scripts/04_revision_ia/etiquetar_por_reglas.py --dry-run
python scripts/04_revision_ia/etiquetar_por_reglas.py
```

Entrada y salida:

```text
data/processed/dataset_consumidores_peru_etiquetado_final.csv   (actualizado en sitio)
reports/03b_reglas/reporte_etiquetado_reglas.csv
reports/03b_reglas/distribucion_final_reglas.csv
reports/03b_reglas/validacion_reglas.csv
```

Reglas conservadoras:

```text
R1: 4/5 estrellas + modelo NEU con prob_neu >= 0.60  -> neutral
R2: 1 estrella  + modelo NEU + prob_pos < 0.25       -> muy negativo
    2 estrellas + modelo NEU + prob_pos < 0.25       -> negativo
```

Recupera 390 etiquetas (343 neutral, 47 negativas) con 93% de acuerdo frente a las etiquetas IA
existentes. Origen marcado como `regla_voto_ponderado`. Residuo sin etiqueta: 620.

## 06. Split Estratificado

Comando:

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

El modelado se divide en tres familias (fases 07-09) y una comparacion final (fase 10).
Las tres familias parten de los mismos splits y se evaluan con el mismo modulo comun
(`scripts/_comun/evaluacion.py`), por lo que sus metricas son directamente comparables.

## 07. Modelos Clasicos (SVM y Naive Bayes)

Comando:

```bash
python scripts/07_modelado/entrenar_evaluar.py
```

Entrada: `data/splits/{train,valid,test}.csv` (columna `texto_modelo`). Salidas:

```text
reports/06_modelado/comparacion_modelos.csv
reports/06_modelado/f1_por_clase.csv
reports/06_modelado/matriz_confusion_<modelo>_<estrategia>_test.csv / .png
reports/06_modelado/reporte_clasificacion_<modelo>_<estrategia>_test.txt
models/mejor_modelo.joblib
```

TF-IDF (1-2 gramas, stopwords ES) + los dos clasicos del enunciado: SVM (LinearSVC) y
Naive Bayes (MultinomialNB), cruzados con base / balanced / SMOTE. Naive Bayes omite
'balanced'. Mejores resultados (F1-Macro en test): naive_bayes+smote 0.5207, svm+balanced
0.5203. SMOTE rescata a Naive Bayes (0.31 -> 0.52), coherente con la literatura del documento.

## 08. Deep Learning (CNN y LSTM)

Comando:

```bash
python scripts/08_dl/entrenar_dl.py
```

Entrada: los mismos splits (`texto_modelo`). Salidas:

```text
reports/07_dl/comparacion_dl.csv
reports/07_dl/f1_por_clase_dl.csv
reports/07_dl/matriz_confusion_<modelo>_<estrategia>_test.csv / .png
reports/07_dl/reporte_clasificacion_<modelo>_<estrategia>_test.txt
models/mejor_modelo_dl.pt
```

PyTorch con embeddings entrenados desde cero: TextCNN (kernels 3/4/5) y BiLSTM, con
estrategias base / class_weight (perdida ponderada). GPU si esta disponible, early stopping
por F1-macro. Mejor: lstm+class_weight 0.5247 en test (a la par de los clasicos).

## 09. Transformers (BETO y XLM-RoBERTa)

Comando:

```bash
python scripts/09_transformers/entrenar_transformers.py
```

Entrada: los mismos splits, pero con `comentario_limpio` (texto natural). Salidas:

```text
reports/08_transformers/comparacion_transformers.csv
reports/08_transformers/f1_por_clase_transformers.csv
reports/08_transformers/matriz_confusion_<modelo>_<estrategia>_test.csv / .png
reports/08_transformers/reporte_clasificacion_<modelo>_<estrategia>_test.txt
models/mejor_modelo_transformer/
```

Fine-tuning de BETO (dccuchile/bert-base-spanish-wwm-cased) y XLM-RoBERTa (xlm-roberta-base)
con HuggingFace, perdida ponderada, AdamW + warmup, AMP en GPU y early stopping por F1-macro.
La primera corrida descarga los modelos. Los transformers son la mejor familia (ver fase 10).

## 10. Comparacion Unificada

Comando:

```bash
python scripts/10_comparacion/comparar_todos.py
```

Une las comparaciones de las fases 07/08/09 en una tabla y un grafico:

```text
reports/09_comparacion/comparacion_global.csv
reports/09_comparacion/comparacion_global_test.csv
reports/09_comparacion/comparacion_global_f1_macro.png
```
