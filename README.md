# Analisis De Sentimiento Multiclase Sobre Resenas De Empresas Peruanas

Proyecto final para construir un clasificador multiclase de sentimiento sobre resenas de consumidores peruanos. El flujo oficial parte del scraping, prepara el dataset, usa una semilla manual de 500 registros, compara seis modelos, amplia el etiquetado del conjunto de desarrollo y evalua el mejor modelo en un 20% de prueba reservado.

## Flujo Final Oficial

```text
1. Scraping de resenas
        ↓
2. Limpieza y normalizacion del dataset
        ↓
3. Dataset base procesado
        ↓
4. Split 80% desarrollo / 20% prueba final
        ↓
5. Del 80% se extrae una semilla de 500 registros
        ↓
6. Etiquetado manual de la semilla de 500 registros
        ↓
7. Validacion cruzada con los 6 modelos
        ↓
8. Seleccion de BETO como mejor modelo inicial
        ↓
9. Reentrenamiento de BETO con la semilla
        ↓
10. Etiquetado automatico del resto del 80%
        ↓
11. Clustering para detectar casos dudosos
        ↓
12. Revision manual de casos dudosos
        ↓
13. Consolidacion del 80% etiquetado
        ↓
14. Division interna 70/30 train-valid
        ↓
15. Entrenamiento final de los 6 modelos
        ↓
16. Seleccion de BETO como modelo final
        ↓
17. Evaluacion unica de BETO en el 20% de prueba
```

## Modelos Evaluados

Se comparan seis modelos en tres familias:

| Familia | Modelos | Representacion |
|---|---|---|
| Clasicos | SVM, Naive Bayes | TF-IDF |
| Deep learning | CNN, LSTM | Embeddings entrenables |
| Transformers | BETO, XLM-RoBERTa | Tokenizacion subpalabra HuggingFace |

La metrica principal es `F1-macro`, porque el problema tiene cinco clases y distribucion desbalanceada.

## Estructura Del Proyecto

```text
data/
  raw/
    empresas.csv
    dataset_consumidores_peru.csv
  processed/
    dataset_consumidores_peru_limpio.csv
    dataset_consumidores_peru_etiquetado_final.csv
  splits_v2/
    test.csv
    semilla.csv
    semilla_etiquetada.csv
    dev_resto.csv
    dev_etiquetado_completo.csv
    test_etiquetado.csv

modelo_entrenado/
  beto_sentimiento/              modelo BETO final versionado con Git LFS

reports/
  10_particion_semilla/          split 80/20 y semilla de 500
  11_etiquetado_manual/          revision y distribucion de la semilla manual
  12_cv_modelos/                 validacion cruzada e hiperparametros
  13_self_training/              etiquetado del resto, clustering y revision
  14_entrenamiento_final/        comparacion final, curvas y test

scripts/
  _comun/                        utilidades compartidas
  01_scraping/                   scraping de Google Maps
  02_limpieza/                   limpieza y auditoria
  11_particion_semilla/          split 80/20 y semilla 500
  13_cv_modelos/                 validacion cruzada de los 6 modelos
  14_self_training/              etiquetado del resto y clustering
  15_entrenamiento_final/        entrenamiento final y evaluacion en test
```

Las carpetas `scripts/07_modelado/`, `scripts/08_dl/` y `scripts/09_transformers/` se conservan como implementaciones base y dependencias reutilizadas por los entrenadores del flujo final.

## Datos Y Particiones

El dataset utilizable contiene 4797 registros.

| Conjunto | Cantidad | Uso |
|---|---:|---|
| Semilla manual | 500 | Validacion cruzada y entrenamiento del modelo inicial |
| Resto desarrollo | 3337 | Etiquetado automatico por el modelo inicial |
| Desarrollo completo | 3837 | Entrenamiento final, luego dividido 70/30 |
| Prueba final | 960 | Evaluacion unica del mejor modelo |

La separacion inicial es 80% desarrollo y 20% prueba final. La seleccion se estratifica por `estrellas x tercil_longitud` para preservar variedad por intensidad de calificacion y longitud del comentario.

## Detalle Fase Por Fase

| Fase | Objetivo | Entrada principal | Salida principal | Evidencia |
|---|---|---|---|---|
| 1. Scraping | Extraer resenas y metadatos de Google Maps. | `data/raw/empresas.csv` | `data/raw/dataset_consumidores_peru.csv` | Dataset crudo versionado. |
| 2. Limpieza | Normalizar texto, validar estrellas y construir columnas de modelado. | `data/raw/dataset_consumidores_peru.csv` | `data/processed/dataset_consumidores_peru_limpio.csv` | Dataset limpio. |
| 3. Dataset base | Mantener el dataset procesado usado por el flujo final. | Dataset limpio/procesado | `data/processed/dataset_consumidores_peru_etiquetado_final.csv` | Dataset base final. |
| 4. Particion 80/20 | Reservar 20% como prueba final y trabajar solo con el 80% restante. | `dataset_consumidores_peru_etiquetado_final.csv` | `data/splits_v2/test.csv`, `semilla.csv`, `dev_resto.csv` | `reports/10_particion_semilla/reporte_particion.csv` |
| 5. Semilla manual | Disponer de 500 registros etiquetados como referencia inicial. | `data/splits_v2/semilla.csv` | `data/splits_v2/semilla_etiquetada.csv` | `reports/11_etiquetado_manual/distribucion_semilla_500.csv` |
| 6. Validacion cruzada | Comparar los seis modelos sobre la semilla de 500. | `semilla_etiquetada.csv` | `mejores_hiperparametros.json` | `reports/12_cv_modelos/cv_resultados.csv` |
| 7. Modelo inicial | Elegir el mejor modelo para ampliar etiquetas. | Resultados CV | BETO seleccionado | `reports/12_cv_modelos/cv_mejor_por_modelo.csv` |
| 8. Self-training | Reentrenar BETO con la semilla y etiquetar el resto del 80%. | `semilla_etiquetada.csv`, `dev_resto.csv` | Etiquetas generadas para desarrollo | `reports/13_self_training/resumen_self_training.csv` |
| 9. Clustering | Detectar inconsistencias y casos dudosos por coherencia de grupos. | Desarrollo etiquetado por modelo | `casos_dudosos_revision.csv` | `reports/13_self_training/pureza_clusters.csv` |
| 10. Revision | Permitir correccion manual de casos dudosos. | `casos_dudosos_revision.csv` | `cambios_revision.csv` | `reports/13_self_training/cambios_revision.csv` |
| 11. Consolidacion | Unir semilla y resto etiquetado en el 80% final. | Semilla + desarrollo etiquetado | `dev_etiquetado_completo.csv` | `reports/13_self_training/distribucion_final_dev.csv` |
| 12. Entrenamiento final | Dividir el 80% en 70/30, entrenar los seis modelos y elegir el mejor. | `dev_etiquetado_completo.csv` | Comparacion final | `reports/14_entrenamiento_final/comparacion_valid.csv` |
| 13. Test final | Evaluar una sola vez el mejor modelo en el 20% reservado. | `test_etiquetado.csv` | Metricas test de BETO | `reports/14_entrenamiento_final/resumen_final.csv` |

## Logica Metodologica

El 20% de prueba se separa al inicio y se mantiene reservado hasta el final. Esto evita que el conjunto de prueba influya en la seleccion de hiperparametros o en la eleccion del modelo. Toda la seleccion ocurre dentro del 80% de desarrollo.

La semilla manual de 500 registros funciona como el primer conjunto supervisado confiable. Sobre esta semilla se aplica validacion cruzada estratificada para comparar seis modelos bajo la misma metrica. El modelo con mejor F1-macro promedio fue BETO, por lo que se uso como etiquetador inicial para ampliar el conjunto de desarrollo.

El clustering no reemplaza a la validacion supervisada. Se usa como control complementario de coherencia: agrupa comentarios parecidos y permite detectar registros cuya etiqueta no coincide con la tendencia de su cluster, especialmente cuando el grupo tiene una clase mayoritaria clara. Esos registros se exportan para revision manual.

Despues de consolidar el 80% etiquetado, se realiza una division interna 70/30 para entrenamiento y validacion final. Con esa division se entrenan nuevamente los seis modelos y se elige el mejor por F1-macro en validacion. Solo ese modelo ganador se evalua en el 20% de prueba.

## Comandos Principales

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar fases principales:

```bash
python scripts/01_scraping/scriptscraping.py
python scripts/02_limpieza/limpiar_dataset.py
python scripts/11_particion_semilla/preparar_particion_semilla.py
python scripts/13_cv_modelos/validacion_cruzada.py
python scripts/14_self_training/self_training.py
python scripts/15_entrenamiento_final/entrenar_final.py
```

Tambien se puede listar la disponibilidad de fases con el orquestador:

```bash
python run_pipeline.py --listar
```

Y ejecutar una fase especifica:

```bash
python run_pipeline.py --solo 13_validacion_cruzada
```

Notas:

- El scraping requiere Chrome configurado para depuracion remota.
- Las fases con transformers pueden requerir GPU para ejecutarse en tiempos razonables.
- La semilla de 500 registros ya se encuentra consolidada en `data/splits_v2/semilla_etiquetada.csv` con la etiqueta objetivo `sentimiento_v2`.

## Hiperparametros Seleccionados

Los hiperparametros se seleccionaron mediante validacion cruzada estratificada de 5 folds sobre la semilla manual de 500 registros.

| Modelo | Hiperparametros seleccionados |
|---|---|
| SVM | `C=0.5`, `class_weight=balanced`, `max_iter=2000` |
| Naive Bayes | `alpha=1.0`, `SMOTE=True` |
| CNN | `lr=0.0005`, `num_filtros=150` |
| LSTM | `lr=0.0005`, `hidden=256` |
| BETO | `lr=3e-05` |
| XLM-RoBERTa | `lr=3e-05` |

Detalle completo:

```text
reports/12_cv_modelos/mejores_hiperparametros.json
```

## Resultados Principales

### Validacion Cruzada Sobre La Semilla

| Modelo | F1-macro CV |
|---|---:|
| BETO | 0.6360 |
| LSTM | 0.4693 |
| SVM | 0.4564 |
| XLM-RoBERTa | 0.4550 |
| CNN | 0.4513 |
| Naive Bayes | 0.4495 |

La validacion cruzada muestra que BETO fue el mejor punto de partida para etiquetar el resto del 80% de desarrollo.

### Entrenamiento Final

En validacion final, BETO fue el mejor modelo:

| Modelo | F1-macro validacion | Accuracy validacion |
|---|---:|---:|
| BETO | 0.7955 | 0.8099 |
| XLM-RoBERTa | 0.7331 | 0.7535 |
| LSTM | 0.6015 | 0.6233 |
| SVM | 0.5972 | 0.6319 |
| CNN | 0.5828 | 0.6155 |
| Naive Bayes | 0.5801 | 0.6163 |

Evaluacion unica de BETO en test:

| Metrica | Valor |
|---|---:|
| F1-macro | 0.6688 |
| Accuracy | 0.6760 |
| Balanced accuracy | 0.6756 |
| F1 weighted | 0.6719 |

Por clase, el modelo obtiene mejor desempeno en `muy negativo` y mayor dificultad en `neutral`, debido a la ambiguedad de comentarios mixtos o de baja polaridad.

### Interpretacion

Los resultados son favorables para un problema de cinco clases, especialmente porque las clases intermedias son cercanas entre si. El F1-macro de test resume el desempeno balanceado entre clases y evita que la evaluacion dependa solo de las clases mas frecuentes. La clase `neutral` es la principal limitacion, porque muchos comentarios combinan elogios y quejas o expresan opiniones de baja intensidad.

## Curvas Y Analisis De Error

El entrenamiento final genera curvas para observar el comportamiento de los modelos:

```text
reports/14_entrenamiento_final/curvas_entrenamiento.csv
reports/14_entrenamiento_final/curva_aprendizaje_svm.png
reports/14_entrenamiento_final/curva_aprendizaje_naive_bayes.png
reports/14_entrenamiento_final/curva_entrenamiento_cnn.png
reports/14_entrenamiento_final/curva_entrenamiento_lstm.png
reports/14_entrenamiento_final/curva_entrenamiento_beto.png
reports/14_entrenamiento_final/curva_entrenamiento_xlm_roberta.png
```

Para SVM y Naive Bayes se grafican curvas de aprendizaje, es decir, F1-macro de validacion frente al numero de ejemplos de entrenamiento. Para CNN, LSTM y transformers se grafican curvas por epoca con perdida de entrenamiento y F1-macro de validacion.

La matriz de confusion y el reporte de clasificacion del modelo final estan en:

```text
reports/14_entrenamiento_final/matriz_confusion_beto_final_test.csv
reports/14_entrenamiento_final/reporte_clasificacion_beto_final_test.txt
```

## Modelo Entrenado

El modelo ganador BETO esta versionado en:

```text
modelo_entrenado/beto_sentimiento/
```

Esta carpeta contiene los archivos necesarios para cargar el modelo con HuggingFace Transformers y usarlo en prediccion o en una interfaz de demostracion.

## Evidencias

Archivos principales de resultados:

```text
reports/10_particion_semilla/reporte_particion.csv
reports/11_etiquetado_manual/distribucion_semilla_500.csv
reports/12_cv_modelos/cv_resultados.csv
reports/12_cv_modelos/cv_mejor_por_modelo.csv
reports/13_self_training/pureza_clusters.csv
reports/13_self_training/casos_dudosos_revision.csv
reports/14_entrenamiento_final/comparacion_valid.csv
reports/14_entrenamiento_final/resumen_final.csv
reports/14_entrenamiento_final/reporte_clasificacion_beto_final_test.txt
reports/14_entrenamiento_final/curvas_entrenamiento.csv
```

## Pruebas

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```
