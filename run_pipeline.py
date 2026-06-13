"""Orquestador del pipeline: corre todas las etapas con un solo comando.

En lugar de ejecutar 6+ comandos a mano, este script encadena las etapas en orden,
verifica que exista la entrada de cada una, salta con un mensaje claro las que no
puedan correr (falta de archivo de entrada, falta de paquete, etc.) y muestra un
resumen final.

Uso tipico:

    python run_pipeline.py                 # corre el procesamiento 02 -> 06
    python run_pipeline.py --listar        # muestra las etapas y si pueden correr
    python run_pipeline.py --dry-run       # muestra que haria, sin ejecutar nada
    python run_pipeline.py --con-scraping  # incluye la etapa 01 (necesita Chrome)
    python run_pipeline.py --desde 05b_reglas          # corre desde una etapa
    python run_pipeline.py --solo 05b_reglas 06_split  # corre solo esas etapas
    python run_pipeline.py --instalar-deps # pip install -r requirements.txt antes

El scraping (01) esta excluido por defecto porque necesita que abras Chrome con
depuracion remota a mano. Inclulo con --con-scraping.
"""

import argparse
import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PY = sys.executable  # mismo interprete que ejecuta este script

# Archivos clave usados para decidir si una etapa tiene su entrada lista.
RAW_EMPRESAS = PROJECT_ROOT / "data" / "raw" / "empresas.csv"
RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "dataset_consumidores_peru.csv"
LIMPIO = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_limpio.csv"
ETIQUETADO = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado.csv"
FINAL = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado_final.csv"
REV_PRIO = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_prioritaria_etiquetas.csv"
REV_PRIO_IA = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_prioritaria_etiquetas_ia.csv"
REV_NEU = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_neutral_para_ia.csv"
REV_NEU_IA = PROJECT_ROOT / "reports" / "03_revision_ia" / "revision_neutral_etiquetas_ia.csv"


@dataclass
class Etapa:
    clave: str                       # id corto para --desde/--hasta/--solo
    titulo: str
    script: Path
    entradas: list = field(default_factory=list)   # rutas que deben existir
    requiere_paquete: str = ""       # paquete python necesario (ej. pysentimiento)
    requiere_chrome: bool = False
    critica: bool = True             # si falla, detiene el pipeline (salvo --continuar-en-error)
    por_defecto: bool = True         # entra en la corrida por defecto


# Orden LOGICO de ejecucion (no por carpeta). El etiquetado por reglas corre
# despues de la integracion IA, ya que actualiza el mismo dataset final.
ETAPAS = [
    Etapa(
        "01_scraping",
        "Scraping de Google Maps",
        PROJECT_ROOT / "scripts" / "01_scraping" / "scriptscraping.py",
        entradas=[RAW_EMPRESAS],
        requiere_paquete="selenium",
        requiere_chrome=True,
        critica=False,
        por_defecto=False,
    ),
    Etapa(
        "02_limpieza",
        "Limpieza del dataset",
        PROJECT_ROOT / "scripts" / "02_limpieza" / "limpiar_dataset.py",
        entradas=[RAW_DATASET],
    ),
    Etapa(
        "02_auditoria",
        "Auditoria del dataset (informativa)",
        PROJECT_ROOT / "scripts" / "02_limpieza" / "auditar_dataset.py",
        entradas=[RAW_DATASET, RAW_EMPRESAS],
        critica=False,
    ),
    Etapa(
        "03_autoetiquetado",
        "Autoetiquetado con pysentimiento",
        PROJECT_ROOT / "scripts" / "03_autoetiquetado" / "autoetiquetar_sentimiento.py",
        entradas=[LIMPIO],
        requiere_paquete="pysentimiento",
    ),
    Etapa(
        "05_integracion_ia",
        "Integracion de etiquetas IA (clases minoritarias)",
        PROJECT_ROOT / "scripts" / "05_integracion" / "integrar_etiquetas_ia.py",
        entradas=[ETIQUETADO, REV_PRIO, REV_PRIO_IA],
        critica=False,
    ),
    Etapa(
        "05_integracion_neutral",
        "Integracion de etiquetas IA (neutral)",
        PROJECT_ROOT / "scripts" / "05_integracion" / "integrar_etiquetas_ia_neutral.py",
        entradas=[FINAL, REV_NEU, REV_NEU_IA],
        critica=False,
    ),
    Etapa(
        "05b_reglas",
        "Etiquetado automatico por reglas",
        PROJECT_ROOT / "scripts" / "04_revision_ia" / "etiquetar_por_reglas.py",
        entradas=[FINAL],
    ),
    Etapa(
        "06_split",
        "Split estratificado train/valid/test",
        PROJECT_ROOT / "scripts" / "06_split" / "preparar_split_dataset.py",
        entradas=[FINAL],
    ),
]

CLAVES = [etapa.clave for etapa in ETAPAS]


def paquete_disponible(nombre):
    if not nombre:
        return True
    return importlib.util.find_spec(nombre) is not None


def motivo_para_saltar(etapa):
    """Devuelve un texto si la etapa NO puede correr ahora, o None si si puede."""
    faltantes = [str(ruta.relative_to(PROJECT_ROOT)) for ruta in etapa.entradas if not ruta.exists()]
    if faltantes:
        return f"falta entrada: {', '.join(faltantes)}"
    if not paquete_disponible(etapa.requiere_paquete):
        return f"falta el paquete '{etapa.requiere_paquete}' (usa --instalar-deps o pip install -r requirements.txt)"
    return None


def seleccionar_etapas(args):
    """Aplica --solo / --desde / --hasta / --con-scraping a la lista de etapas."""
    if args.solo:
        seleccion = [e for e in ETAPAS if e.clave in args.solo]
    else:
        seleccion = [e for e in ETAPAS if e.por_defecto or (e.clave == "01_scraping" and args.con_scraping)]

    if args.desde:
        inicio = CLAVES.index(args.desde)
        seleccion = [e for e in seleccion if CLAVES.index(e.clave) >= inicio]
    if args.hasta:
        fin = CLAVES.index(args.hasta)
        seleccion = [e for e in seleccion if CLAVES.index(e.clave) <= fin]

    return seleccion


def instalar_dependencias():
    requirements = PROJECT_ROOT / "requirements.txt"
    print(f">> Instalando dependencias desde {requirements} ...")
    resultado = subprocess.run([PY, "-m", "pip", "install", "-r", str(requirements)])
    if resultado.returncode != 0:
        print("!! Fallo la instalacion de dependencias.")
        return False
    return True


def correr_etapa(etapa):
    """Ejecuta el script de la etapa. Devuelve el return code."""
    print("\n" + "=" * 78)
    print(f">> [{etapa.clave}] {etapa.titulo}")
    if etapa.requiere_chrome:
        print("   (requiere Chrome abierto con --remote-debugging-port=9222)")
    print(f"   script: {etapa.script.relative_to(PROJECT_ROOT)}")
    print("=" * 78)
    inicio = time.time()
    resultado = subprocess.run([PY, str(etapa.script)], cwd=str(PROJECT_ROOT))
    segundos = time.time() - inicio
    print(f"   -> termino en {segundos:.1f}s con codigo {resultado.returncode}")
    return resultado.returncode


def listar(seleccion):
    print("Etapas del pipeline (orden de ejecucion):\n")
    for etapa in ETAPAS:
        en_seleccion = "*" if etapa in seleccion else " "
        motivo = motivo_para_saltar(etapa)
        estado = "LISTA" if motivo is None else f"NO LISTA ({motivo})"
        print(f"[{en_seleccion}] {etapa.clave:<24} {estado}")
    print("\n* = entra en la corrida actual. Usa --solo/--desde/--hasta para acotar.")


def main():
    args = obtener_argumentos()

    if args.instalar_deps and not instalar_dependencias():
        sys.exit(1)

    seleccion = seleccionar_etapas(args)

    if args.listar:
        listar(seleccion)
        return

    if not seleccion:
        print("No hay etapas seleccionadas. Revisa --solo/--desde/--hasta.")
        return

    print(f"Pipeline: se ejecutaran {len(seleccion)} etapa(s): {', '.join(e.clave for e in seleccion)}")
    if args.dry_run:
        print("\n=== DRY-RUN: no se ejecuta nada ===")
        for etapa in seleccion:
            motivo = motivo_para_saltar(etapa)
            estado = "correria" if motivo is None else f"SALTARIA ({motivo})"
            print(f"- {etapa.clave:<24} {estado}")
        return

    resumen = []
    for etapa in seleccion:
        motivo = motivo_para_saltar(etapa)
        if motivo is not None:
            print(f"\n>> [{etapa.clave}] SALTADA: {motivo}")
            resumen.append((etapa.clave, "SALTADA", motivo))
            continue

        codigo = correr_etapa(etapa)
        if codigo == 0:
            resumen.append((etapa.clave, "OK", ""))
        else:
            resumen.append((etapa.clave, "FALLO", f"codigo {codigo}"))
            if etapa.critica and not args.continuar_en_error:
                print(f"\n!! Etapa critica '{etapa.clave}' fallo. Se detiene el pipeline.")
                print("   (usa --continuar-en-error para seguir de todos modos)")
                break

    print("\n" + "=" * 78)
    print("RESUMEN DEL PIPELINE")
    print("=" * 78)
    for clave, estado, detalle in resumen:
        linea = f"{estado:<8} {clave}"
        if detalle:
            linea += f"  ({detalle})"
        print(linea)

    hubo_fallo = any(estado == "FALLO" for _, estado, _ in resumen)
    sys.exit(1 if hubo_fallo else 0)


def obtener_argumentos():
    parser = argparse.ArgumentParser(
        description="Orquesta el pipeline completo con un solo comando.",
    )
    parser.add_argument("--listar", action="store_true", help="Muestra las etapas y si pueden correr.")
    parser.add_argument("--dry-run", action="store_true", help="Muestra que se haria, sin ejecutar.")
    parser.add_argument("--con-scraping", action="store_true", help="Incluye la etapa 01 de scraping.")
    parser.add_argument("--solo", nargs="+", choices=CLAVES, help="Corre solo estas etapas.")
    parser.add_argument("--desde", choices=CLAVES, help="Corre desde esta etapa en adelante.")
    parser.add_argument("--hasta", choices=CLAVES, help="Corre hasta esta etapa (inclusive).")
    parser.add_argument("--instalar-deps", action="store_true", help="pip install -r requirements.txt antes de correr.")
    parser.add_argument("--continuar-en-error", action="store_true", help="No detenerse si una etapa critica falla.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
