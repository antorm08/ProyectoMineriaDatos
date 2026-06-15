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

## 07. Entrenamiento Y Evaluacion

Comando:

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

Clasificador: TF-IDF (1-2 gramas) + Regresion Logistica. Compara tres estrategias de balanceo
(base, class_weight='balanced', SMOTE en train). Metrica principal F1-Macro.

Resultados actuales (F1-Macro):

```text
estrategia   valid    test
base         0.3834   0.4040
balanced     0.5048   0.5665   (mejor)
smote        0.4789   0.5517
```
