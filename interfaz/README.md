# Interfaz Streamlit - Sentimix Peru

Interfaz de demostracion para usar el modelo BETO final entrenado en el proyecto.

## Requisitos

Instala dependencias desde la raiz del proyecto:

```bash
pip install -r requirements.txt
```

Si clonaste el repositorio y el modelo esta en Git LFS, descarga los pesos:

```bash
git lfs install
git lfs pull
```

## Ejecutar

Desde la raiz del proyecto:

```bash
streamlit run interfaz/app.py
```

La app carga el modelo desde:

```text
modelo_entrenado/beto_sentimiento/
```

## Funcionalidades

- Analisis individual de una resena.
- Probabilidades por clase.
- Analisis por lote desde CSV.
- Descarga de predicciones.
- Vista de metricas del modelo final.
