# Analisis de Sentimiento Multiclase sobre Resenas de Empresas Peruanas

Este proyecto inicia con un scraper de Google Maps para construir un primer dataset de resenas de empresas peruanas.

## Instalacion

```bash
pip install -r requirements.txt
```

## Abrir Chrome para Selenium

Antes de ejecutar el scraper, abre Chrome con debugging remoto:

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome-debug"
```

Si Chrome esta instalado en `Program Files (x86)`, usa:

```powershell
& "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome-debug"
```

Cuando se abra Chrome, puedes elegir `Mantener sesion cerrada`.

## Ejecutar Scraper

```bash
python scripts/01_scraping/scriptscraping.py
```

El scraper lee los negocios desde:

```text
data/raw/empresas.csv
```

Ese archivo debe mantener estas columnas:

```text
nombre,sede,rubro,url
```

Puedes dejar URLs vacias temporalmente:

```csv
BCP,Miraflores,Banca,""
```

El scraper omitira esas filas hasta que pegues el link de Google Maps.

Para cambiar la cantidad maxima de resenas por empresa:

```bash
python scripts/01_scraping/scriptscraping.py --max-reviews 150
```

Para cambiar el archivo de salida:

```bash
python scripts/01_scraping/scriptscraping.py --max-reviews 150 --output data/raw/dataset_consumidores_peru.csv
```

Para usar otro archivo de empresas:

```bash
python scripts/01_scraping/scriptscraping.py --empresas data/raw/empresas.csv --max-reviews 150
```

Para probar solo algunas sedes:

```bash
python scripts/01_scraping/scriptscraping.py --max-reviews 20 --limit-companies 3 --output data/raw/prueba_3_sedes.csv
```

El archivo generado sera:

```text
data/raw/dataset_consumidores_peru.csv
```

## Estructura Del Proyecto

- `scripts/`: scripts organizados por fase numerada del pipeline.
- `data/raw/`: datos originales o extraidos sin transformar.
- `data/processed/`: datasets limpios o enriquecidos.
- `data/splits/`: particiones de entrenamiento, validacion y prueba.
- `reports/`: reportes reproducibles y distribuciones de clases.
- `PIPELINE.md`: orden completo de ejecucion y salidas por fase.

## Limpieza Y Auditoria

```bash
python scripts/02_limpieza/limpiar_dataset.py
python scripts/02_limpieza/auditar_dataset.py
```

La limpieza genera:

```text
data/processed/dataset_consumidores_peru_limpio.csv
reports/01_limpieza/reporte_limpieza_dataset.csv
reports/01_limpieza/distribucion_sentimiento.csv
reports/01_limpieza/distribucion_estrellas.csv
```

## Autoetiquetado

Instala dependencias:

```bash
pip install -r requirements.txt
```

Ejecuta el autoetiquetado:

```bash
python scripts/03_autoetiquetado/autoetiquetar_sentimiento.py
```

El autoetiquetado genera:

```text
data/processed/dataset_consumidores_peru_etiquetado.csv
reports/02_autoetiquetado/reporte_autoetiquetado.csv
reports/02_autoetiquetado/distribucion_sentimiento_final.csv
```

## Columnas Del Dataset

- `comentario`
- `empresa`
- `sede`
- `rubro`
- `estrellas`
- `sentimiento_estrella`
- `fecha_resena`
- `url`

## Etiquetas Iniciales

- 1 estrella: `muy negativo`
- 2 estrellas: `negativo`
- 3 estrellas: `neutral`
- 4 estrellas: `positivo`
- 5 estrellas: `muy positivo`

## Que Hace El Scraper

- Lee la lista de lugares desde `data/raw/empresas.csv`.
- Abre cada URL de Google Maps.
- Intenta abrir el panel de resenas.
- Hace scroll hasta cargar resenas.
- Extrae comentario, estrellas y fecha textual.
- Guarda resultados acumulados en CSV.
- Elimina duplicados por comentario, empresa y sede.

## Que Debes Hacer Tu

- Revisar que cada URL de `data/raw/empresas.csv` apunte al local correcto.
- Agregar mas filas para tener mas empresas y sedes.
- Completar las filas que tienen `""` en la columna `url`.
- Mantener los rubros consistentes, por ejemplo `Banca`, `Retail`, `Farmacia`, `Telecomunicaciones`.
- Abrir Chrome con debugging remoto antes de ejecutar el scraper.
- Revisar el CSV generado para detectar negocios que no devolvieron resenas.

## Que Puedo Hacer Yo

- Mejorar el codigo del scraper.
- Agregar validaciones y columnas utiles.
- Adaptar el scraper si Google Maps cambia algun selector.
- Crear scripts de limpieza, analisis exploratorio y entrenamiento NLP.
- Ayudarte a revisar errores despues de una ejecucion.
