"""Configuracion de pytest: agrega las carpetas de scripts al sys.path.

Permite importar tanto los modulos comunes (`_comun.*`) como los scripts puntuales
(etiquetar_por_reglas, autoetiquetar_sentimiento) sin instalar el proyecto como paquete.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for subcarpeta in ["scripts", "scripts/03_autoetiquetado", "scripts/04_revision_ia"]:
    sys.path.insert(0, str(ROOT / subcarpeta))
