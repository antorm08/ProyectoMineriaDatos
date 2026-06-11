# Analisis De Sentimiento Multiclase Sobre Resenas De Empresas Peruanas

Proyecto para construir, limpiar, etiquetar y preparar un dataset de resenas de consumidores peruanos para clasificacion de sentimiento multiclase.

El flujo actual no depende solo de estrellas de Google. Usa una estrategia por etapas:

```text
scraping
limpieza
autoetiquetado con pysentimiento
revision asistida por IA de casos ambiguos
integracion conservadora de etiquetas
split estratificado
entrenamiento y evaluacion
```

## Instalacion

```bash
pip install -r requirements.txt
```

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
muy negativo    1018
positivo         647
negativo         422
neutral          296
```

Total con etiqueta final consolidada:

```text
3790
```

Casos sin etiqueta final confiable:

```text
1010
```

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

### 06. Split Estratificado

Pendiente de implementar.

Salidas esperadas:

```text
data/splits/train.csv
data/splits/valid.csv
data/splits/test.csv
reports/05_split/reporte_split_dataset.csv
reports/05_split/distribucion_split_train.csv
reports/05_split/distribucion_split_valid.csv
reports/05_split/distribucion_split_test.csv
```

### 07. Entrenamiento Y Evaluacion

Pendiente de implementar.

Comparaciones recomendadas:

```text
modelo base sin balanceo
modelo con class_weight='balanced'
modelo con SMOTE solo en train
```

Metricas principales:

```text
F1-Macro
F1 por clase
matriz de confusion
accuracy como metrica secundaria
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
