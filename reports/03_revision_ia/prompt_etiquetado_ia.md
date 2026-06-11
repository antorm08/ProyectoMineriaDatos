# Prompt Para Etiquetado Asistido Por IA

Usar este prompt junto con el archivo:

```text
reports/03_revision_ia/revision_prioritaria_para_ia.csv
```

## Prompt

```text
Actúa como anotador de datos para un proyecto académico de análisis de sentimiento multiclase sobre reseñas de consumidores peruanos.

Debes clasificar cada reseña en una sola de estas cinco clases:

- muy negativo: queja fuerte, pésima experiencia, enojo claro, pérdida de dinero, maltrato, estafa, insultos o rechazo intenso.
- negativo: experiencia mala, molestia o crítica clara, pero no extrema.
- neutral: opinión mixta, regular, ambigua, informativa, sin emoción clara o con aspectos positivos y negativos equilibrados.
- positivo: buena experiencia, satisfacción o comentario favorable.
- muy positivo: elogio fuerte, entusiasmo claro, recomendación explícita o experiencia excelente.

Usa el texto de la reseña como fuente principal.
Las estrellas, sentimiento_estrella, sentimiento_modelo, probabilidades y sentimiento_final_provisional son solo apoyo, no verdad absoluta.

Devuelve el resultado como un archivo CSV descargable llamado:

revision_prioritaria_etiquetas_ia.csv

El CSV debe tener exactamente estas columnas:

id_revision,sentimiento_ia,confianza_ia,justificacion_ia,usar_etiqueta_ia

Reglas:
- Mantén el mismo id_revision de entrada.
- No cambies el orden de las filas.
- sentimiento_ia debe ser exactamente una de estas clases: muy negativo, negativo, neutral, positivo, muy positivo.
- confianza_ia debe ser exactamente: alta, media o baja.
- usar_etiqueta_ia debe ser exactamente: si o no.
- Usa usar_etiqueta_ia=si cuando el comentario permita decidir razonablemente la clase.
- Usa usar_etiqueta_ia=no cuando el texto sea vacío, demasiado corto, contradictorio o muy ambiguo.
- Si el texto es vacío, demasiado corto o no permite decidir, usa sentimiento_ia=neutral, confianza_ia=baja y usar_etiqueta_ia=no.
- justificacion_ia debe tener máximo 12 palabras.
- No inventes información que no esté en el comentario.
- No agregues columnas extra.
- No incluyas explicaciones fuera del CSV.
- Devuelve solo el archivo CSV solicitado.
```
