"""Orquestador del flujo final del proyecto.

El flujo oficial es:

    01 scraping -> 02 limpieza -> 11 particion/semilla ->
    13 validacion cruzada -> 14 self-training/clustering ->
    15 entrenamiento final

Las fases pesadas no se ejecutan por defecto. Usa --solo para correr una fase
especifica o --listar para ver entradas y disponibilidad.
"""

import argparse
import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PY = sys.executable

RAW_EMPRESAS = PROJECT_ROOT / "data" / "raw" / "empresas.csv"
RAW_DATASET = PROJECT_ROOT / "data" / "raw" / "dataset_consumidores_peru.csv"
LIMPIO = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_limpio.csv"
FINAL = PROJECT_ROOT / "data" / "processed" / "dataset_consumidores_peru_etiquetado_final.csv"
SEMILLA = PROJECT_ROOT / "data" / "splits_v2" / "semilla_etiquetada.csv"
DEV_RESTO = PROJECT_ROOT / "data" / "splits_v2" / "dev_resto.csv"
DEV_COMPLETO = PROJECT_ROOT / "data" / "splits_v2" / "dev_etiquetado_completo.csv"
TEST_ETIQUETADO = PROJECT_ROOT / "data" / "splits_v2" / "test_etiquetado.csv"
CV_JSON = PROJECT_ROOT / "reports" / "12_cv_modelos" / "mejores_hiperparametros.json"


@dataclass
class Etapa:
    clave: str
    titulo: str
    script: Path
    entradas: list = field(default_factory=list)
    requiere_paquete: str = ""
    critica: bool = True
    por_defecto: bool = False


ETAPAS = [
    Etapa(
        "01_scraping",
        "Scraping de Google Maps",
        PROJECT_ROOT / "scripts" / "01_scraping" / "scriptscraping.py",
        entradas=[RAW_EMPRESAS],
        requiere_paquete="selenium",
    ),
    Etapa(
        "02_limpieza",
        "Limpieza del dataset",
        PROJECT_ROOT / "scripts" / "02_limpieza" / "limpiar_dataset.py",
        entradas=[RAW_DATASET],
    ),
    Etapa(
        "11_particion_semilla",
        "Split 80/20 y seleccion de semilla de 500",
        PROJECT_ROOT / "scripts" / "11_particion_semilla" / "preparar_particion_semilla.py",
        entradas=[FINAL],
    ),
    Etapa(
        "13_validacion_cruzada",
        "Validacion cruzada con seis modelos",
        PROJECT_ROOT / "scripts" / "13_cv_modelos" / "validacion_cruzada.py",
        entradas=[SEMILLA],
        requiere_paquete="torch",
    ),
    Etapa(
        "14_self_training",
        "Etiquetado del resto, clustering y revision",
        PROJECT_ROOT / "scripts" / "14_self_training" / "self_training.py",
        entradas=[SEMILLA, DEV_RESTO, CV_JSON],
        requiere_paquete="torch",
    ),
    Etapa(
        "15_entrenamiento_final",
        "Entrenamiento final y evaluacion en test",
        PROJECT_ROOT / "scripts" / "15_entrenamiento_final" / "entrenar_final.py",
        entradas=[DEV_COMPLETO, TEST_ETIQUETADO, CV_JSON],
        requiere_paquete="torch",
    ),
]

CLAVES = [etapa.clave for etapa in ETAPAS]


def paquete_disponible(nombre):
    if not nombre:
        return True
    return importlib.util.find_spec(nombre) is not None


def motivo_para_saltar(etapa):
    faltantes = [str(ruta.relative_to(PROJECT_ROOT)) for ruta in etapa.entradas if not ruta.exists()]
    if faltantes:
        return f"falta entrada: {', '.join(faltantes)}"
    if not paquete_disponible(etapa.requiere_paquete):
        return f"falta paquete '{etapa.requiere_paquete}'"
    return None


def seleccionar_etapas(args):
    seleccion = [e for e in ETAPAS if e.por_defecto]
    if args.solo:
        seleccion = [e for e in ETAPAS if e.clave in args.solo]
    if args.desde:
        inicio = CLAVES.index(args.desde)
        seleccion = [e for e in ETAPAS if CLAVES.index(e.clave) >= inicio]
    if args.hasta:
        fin = CLAVES.index(args.hasta)
        seleccion = [e for e in seleccion if CLAVES.index(e.clave) <= fin]
    return seleccion


def listar(seleccion):
    print("Etapas del flujo final:\n")
    for etapa in ETAPAS:
        marca = "*" if etapa in seleccion else " "
        motivo = motivo_para_saltar(etapa)
        estado = "LISTA" if motivo is None else f"NO LISTA ({motivo})"
        print(f"{marca} {etapa.clave:<24} {estado:<60} {etapa.titulo}")


def correr_etapa(etapa):
    print("\n" + "=" * 78)
    print(f">> [{etapa.clave}] {etapa.titulo}")
    print(f"   script: {etapa.script.relative_to(PROJECT_ROOT)}")
    print("=" * 78)
    inicio = time.time()
    resultado = subprocess.run([PY, str(etapa.script)], cwd=str(PROJECT_ROOT))
    print(f"   -> termino en {time.time() - inicio:.1f}s con codigo {resultado.returncode}")
    return resultado.returncode


def obtener_argumentos():
    parser = argparse.ArgumentParser(description="Orquestador del flujo final.")
    parser.add_argument("--listar", action="store_true", help="Muestra etapas y disponibilidad.")
    parser.add_argument("--dry-run", action="store_true", help="Muestra que correria, sin ejecutar.")
    parser.add_argument("--solo", nargs="+", choices=CLAVES, help="Ejecuta solo las etapas indicadas.")
    parser.add_argument("--desde", choices=CLAVES, help="Ejecuta desde una etapa.")
    parser.add_argument("--hasta", choices=CLAVES, help="Ejecuta hasta una etapa.")
    parser.add_argument("--continuar-en-error", action="store_true")
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    seleccion = seleccionar_etapas(args)
    if args.listar or args.dry_run:
        listar(seleccion)
        return
    if not seleccion:
        print("No hay etapas seleccionadas. Usa --listar o --solo <etapa>.")
        return

    for etapa in seleccion:
        motivo = motivo_para_saltar(etapa)
        if motivo:
            print(f">> Saltando {etapa.clave}: {motivo}")
            if etapa.critica and not args.continuar_en_error:
                raise SystemExit(1)
            continue
        codigo = correr_etapa(etapa)
        if codigo != 0 and etapa.critica and not args.continuar_en_error:
            raise SystemExit(codigo)


if __name__ == "__main__":
    main()
