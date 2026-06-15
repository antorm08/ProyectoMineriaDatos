import argparse
import logging
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from urllib3.exceptions import ReadTimeoutError


DEBUGGER_ADDRESS = "127.0.0.1:9222"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from _comun.texto import colapsar_espacios as normalizar_texto  # noqa: E402

EMPRESAS_FILE = PROJECT_ROOT / "data" / "raw" / "empresas.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "dataset_consumidores_peru.csv"
MAX_REVIEWS_PER_COMPANY = 40
SCROLL_ATTEMPTS = 25
SCROLL_PAUSE_SECONDS = 2
COLUMNAS_EMPRESAS = ["nombre", "sede", "rubro", "url"]
SELENIUM_RECUPERABLE_ERRORS = (WebDriverException, ReadTimeoutError, TimeoutError)

# Anti-bloqueo: delays con jitter, pausas humanas entre sedes y backoff ante bloqueos.
# El jitter evita el patron de intervalos constantes (huella de bot); el backoff frena
# la ejecucion cuando Google muestra una pagina de verificacion, en vez de seguir insistiendo.
DELAY_SCROLL_MIN = 1.5
DELAY_SCROLL_MAX = 3.0
PAUSA_SEDE_MIN = 4.0
PAUSA_SEDE_MAX = 10.0
MAX_BLOQUEOS_CONSECUTIVOS = 3
BACKOFF_BASE_SEGUNDOS = 30
BACKOFF_MAX_SEGUNDOS = 300
SENALES_BLOQUEO = ("unusual traffic", "trafico inusual", "tráfico inusual")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def dormir(minimo, maximo):
    """Pausa una duracion aleatoria (jitter) para imitar el ritmo de un humano."""
    if maximo < minimo:
        maximo = minimo
    time.sleep(random.uniform(minimo, maximo))


def pagina_bloqueada(driver):
    """Detecta si Google esta mostrando una pagina de bloqueo / verificacion."""
    try:
        url = (driver.current_url or "").lower()
        fuente = (driver.page_source or "").lower()
    except SELENIUM_RECUPERABLE_ERRORS:
        return False

    if "/sorry/" in url:
        return True
    return any(senal in fuente for senal in SENALES_BLOQUEO)


def esperar_si_bloqueado(driver, bloqueos_consecutivos):
    """Si detecta bloqueo, espera con backoff exponencial. Devuelve el contador actualizado.

    Sin bloqueo: devuelve 0 (reinicia el contador). Con bloqueo: incrementa el contador y
    pausa BACKOFF_BASE * 2^(n-1) segundos (con tope) antes de continuar.
    """
    if not pagina_bloqueada(driver):
        return 0

    bloqueos_consecutivos += 1
    espera = min(BACKOFF_MAX_SEGUNDOS, BACKOFF_BASE_SEGUNDOS * (2 ** (bloqueos_consecutivos - 1)))
    logging.warning(
        "Posible bloqueo de Google detectado (%s consecutivo). Pausando %ss antes de continuar.",
        bloqueos_consecutivos,
        int(espera),
    )
    time.sleep(espera)
    return bloqueos_consecutivos


def crear_driver():
    options = Options()
    options.debugger_address = DEBUGGER_ADDRESS
    return webdriver.Chrome(options=options)


def cargar_empresas(empresas_file):
    if not empresas_file.exists():
        raise FileNotFoundError(f"No existe el archivo de empresas: {empresas_file}")

    df = pd.read_csv(empresas_file)
    total_filas = len(df)
    faltantes = [columna for columna in COLUMNAS_EMPRESAS if columna not in df.columns]
    if faltantes:
        raise ValueError(
            f"El archivo {empresas_file} no tiene estas columnas requeridas: {faltantes}"
        )

    df = df[COLUMNAS_EMPRESAS].fillna("")
    for columna in COLUMNAS_EMPRESAS:
        df[columna] = df[columna].astype(str).map(normalizar_texto)

    filas_con_nombre = df["nombre"] != ""
    filas_con_url_google_maps = df["url"].str.startswith("https://www.google.com/maps")
    filas_omitidas = df[filas_con_nombre & ~filas_con_url_google_maps]

    if not filas_omitidas.empty:
        logging.warning(
            "Filas omitidas por URL vacia o invalida en %s: %s",
            empresas_file,
            len(filas_omitidas),
        )

    df = df[filas_con_nombre & filas_con_url_google_maps]
    total_validas_antes_deduplicar = len(df)
    df = df.drop_duplicates(subset=["nombre", "sede", "url"])
    duplicadas = total_validas_antes_deduplicar - len(df)

    if duplicadas:
        logging.warning("Filas duplicadas omitidas en %s: %s", empresas_file, duplicadas)
    logging.info(
        "Filas en empresas: %s | validas: %s | usadas: %s",
        total_filas,
        total_validas_antes_deduplicar,
        len(df),
    )

    empresas = df.to_dict("records")
    if not empresas:
        raise ValueError(f"No hay empresas validas en {empresas_file}")

    return empresas


def sentimiento_por_estrellas(estrellas):
    etiquetas = {
        1: "muy negativo",
        2: "negativo",
        3: "neutral",
        4: "positivo",
        5: "muy positivo",
    }
    return etiquetas.get(estrellas)


def extraer_estrellas(tarjeta):
    try:
        aria = tarjeta.find_element(By.CLASS_NAME, "kvMYJc").get_attribute("aria-label")
    except SELENIUM_RECUPERABLE_ERRORS:
        return None

    if not aria:
        return None

    match = re.search(r"\d", aria)
    return int(match.group()) if match else None


def extraer_fecha(tarjeta):
    posibles_selectores = [
        (By.CLASS_NAME, "rsqaWe"),
        (By.XPATH, './/span[contains(text(), "hace ")]'),
        (By.XPATH, './/span[contains(text(), "ayer")]'),
    ]

    for by, selector in posibles_selectores:
        try:
            fecha = tarjeta.find_element(by, selector).text
        except SELENIUM_RECUPERABLE_ERRORS:
            continue

        fecha = normalizar_texto(fecha)
        if fecha:
            return fecha

    return None


def aceptar_cookies(driver, wait):
    try:
        boton_aceptar = wait.until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(., "Aceptar todo")]'))
        )
        boton_aceptar.click()
        time.sleep(2)
    except TimeoutException:
        logging.info("No aparecio el boton de cookies.")


def abrir_panel_resenas(driver, wait):
    posibles_botones = [
        '//button[contains(@aria-label, "reseñas")]',
        '//button[contains(@aria-label, "opiniones")]',
        '//button[contains(@aria-label, "reviews")]',
        '//button[contains(., "Reseñas")]',
        '//button[contains(., "Opiniones")]',
        '//button[contains(., "Reviews")]',
    ]

    for xpath in posibles_botones:
        try:
            boton_resenas = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
            boton_resenas.click()
            time.sleep(5)
            return True
        except TimeoutException:
            continue

    return False


def encontrar_panel_scroll(driver):
    posibles_scroll = [
        '//div[@role="feed"]',
        '//div[contains(@class, "m6QErb") and .//div[contains(@class, "jftiEf")]]',
    ]
    mejor_panel = None
    mejor_puntaje = -1

    for xpath in posibles_scroll:
        elementos = driver.find_elements(By.XPATH, xpath)
        for elemento in elementos:
            try:
                tarjetas = elemento.find_elements(By.XPATH, './/div[contains(@class, "jftiEf")]')
                scroll_height = driver.execute_script("return arguments[0].scrollHeight;", elemento)
                client_height = driver.execute_script("return arguments[0].clientHeight;", elemento)
                if not elemento.is_displayed() or elemento.size["height"] <= 0 or not tarjetas:
                    continue

                puntaje = len(tarjetas) * 10
                if scroll_height > client_height:
                    puntaje += 5

                if puntaje > mejor_puntaje:
                    mejor_panel = elemento
                    mejor_puntaje = puntaje
            except SELENIUM_RECUPERABLE_ERRORS:
                continue

    return mejor_panel


def esta_en_reviews_de_usuario(driver):
    try:
        url_actual = driver.current_url
    except SELENIUM_RECUPERABLE_ERRORS:
        return False

    return "/maps/contrib/" in url_actual or "/contrib/" in url_actual


def cerrar_pestana_reviews_usuario(driver):
    try:
        handles = driver.window_handles
        handle_actual = driver.current_window_handle
    except SELENIUM_RECUPERABLE_ERRORS:
        return False

    if not esta_en_reviews_de_usuario(driver):
        return False

    logging.warning("Se detecto una pestana de reseñas de usuario. Cerrandola.")

    if len(handles) <= 1:
        try:
            driver.back()
            time.sleep(2)
            return True
        except SELENIUM_RECUPERABLE_ERRORS:
            return False

    try:
        driver.close()
    except SELENIUM_RECUPERABLE_ERRORS:
        return False

    for handle in handles:
        if handle == handle_actual:
            continue

        try:
            driver.switch_to.window(handle)
            if not esta_en_reviews_de_usuario(driver):
                time.sleep(1)
                return True
        except SELENIUM_RECUPERABLE_ERRORS:
            continue

    try:
        driver.switch_to.window(handles[0])
        time.sleep(1)
        return True
    except SELENIUM_RECUPERABLE_ERRORS:
        return False


def recuperar_panel_empresa(driver, empresa):
    cerrar_pestana_reviews_usuario(driver)

    if not esta_en_reviews_de_usuario(driver):
        return encontrar_panel_scroll(driver)

    logging.warning(
        "Google Maps abrio reseñas de usuario. Volviendo a %s - %s.",
        empresa["nombre"],
        empresa["sede"],
    )

    try:
        driver.get(empresa["url"])
        time.sleep(8)
        wait = WebDriverWait(driver, 20)
        if not abrir_panel_resenas(driver, wait):
            return None
        return encontrar_panel_scroll(driver)
    except SELENIUM_RECUPERABLE_ERRORS as error:
        logging.warning("No se pudo recuperar la ficha de empresa: %s", error)
        return None


def contar_tarjetas(panel_scroll):
    return len(panel_scroll.find_elements(By.XPATH, './/div[contains(@class, "jftiEf")]'))


def cargar_resenas(driver, panel_scroll, max_reviews):
    tarjetas_previas = 0
    intentos_sin_cambios = 0

    for intento in range(SCROLL_ATTEMPTS):
        driver.execute_script(
            "arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].clientHeight;",
            panel_scroll,
        )
        dormir(DELAY_SCROLL_MIN, DELAY_SCROLL_MAX)

        tarjetas_actuales = contar_tarjetas(panel_scroll)
        logging.info("Scroll %s/%s | tarjetas cargadas: %s", intento + 1, SCROLL_ATTEMPTS, tarjetas_actuales)

        if tarjetas_actuales >= max_reviews:
            break

        if tarjetas_actuales == tarjetas_previas:
            intentos_sin_cambios += 1
        else:
            intentos_sin_cambios = 0

        if intentos_sin_cambios >= 4:
            break

        tarjetas_previas = tarjetas_actuales


def hacer_scroll_resenas(driver, panel_scroll, empresa):
    cerrar_modal_compartir(driver)

    panel_actual = recuperar_panel_empresa(driver, empresa)
    if panel_actual is not None:
        panel_scroll = panel_actual

    try:
        driver.execute_script(
            "arguments[0].scrollTop = arguments[0].scrollTop + 1500;",
            panel_scroll,
        )
        driver.execute_script(
            "arguments[0].dispatchEvent(new WheelEvent('wheel', "
            "{deltaY: 1500, bubbles: true, cancelable: true}));",
            panel_scroll,
        )
    except SELENIUM_RECUPERABLE_ERRORS as error:
        logging.warning("Scroll por JavaScript omitido por error recuperable: %s", error)

    try:
        tarjetas = panel_scroll.find_elements(By.XPATH, './/div[contains(@class, "jftiEf")]')
        if tarjetas:
            driver.execute_script("arguments[0].scrollIntoView({block: 'end'});", tarjetas[-1])
    except SELENIUM_RECUPERABLE_ERRORS as error:
        logging.warning("Scroll a ultima tarjeta omitido por error recuperable: %s", error)

    cerrar_modal_compartir(driver)
    return panel_scroll


def cerrar_modal_compartir(driver):
    try:
        dialogos = driver.find_elements(
            By.XPATH,
            '//div[@role="dialog" and (contains(., "Compartir") or contains(., "Share"))]',
        )
    except SELENIUM_RECUPERABLE_ERRORS:
        return False

    modal_visible = False
    for dialogo in dialogos:
        try:
            if dialogo.is_displayed():
                modal_visible = True
                break
        except SELENIUM_RECUPERABLE_ERRORS:
            continue

    if not modal_visible:
        return False

    posibles_cierres = [
        '//div[@role="dialog" and (contains(., "Compartir") or contains(., "Share"))]//button[contains(@aria-label, "Cerrar")]',
        '//div[@role="dialog" and (contains(., "Compartir") or contains(., "Share"))]//button[contains(@aria-label, "Close")]',
        '//div[@role="dialog" and (contains(., "Compartir") or contains(., "Share"))]//button[contains(@aria-label, "Volver")]',
        '//div[@role="dialog" and (contains(., "Compartir") or contains(., "Share"))]//button[contains(@aria-label, "Back")]',
        '//div[@role="dialog" and (contains(., "Compartir") or contains(., "Share"))]//button[.="×" or .=","]',
    ]

    for xpath in posibles_cierres:
        for boton in driver.find_elements(By.XPATH, xpath):
            try:
                if boton.is_displayed():
                    driver.execute_script("arguments[0].click();", boton)
                    time.sleep(0.5)
                    return True
            except SELENIUM_RECUPERABLE_ERRORS:
                continue

    try:
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
        time.sleep(0.5)
        return True
    except SELENIUM_RECUPERABLE_ERRORS:
        return False


def recolectar_resenas_con_scroll(driver, panel_scroll, empresa, max_reviews, expand_comments=True):
    resenas_por_clave = {}
    intentos_sin_nuevas = 0
    botones_mas_intentados = set()

    for intento in range(SCROLL_ATTEMPTS):
        cerrar_modal_compartir(driver)
        panel_actual = recuperar_panel_empresa(driver, empresa)
        if panel_actual is not None:
            panel_scroll = panel_actual
        elif esta_en_reviews_de_usuario(driver):
            logging.warning("No se pudo volver al panel de la empresa. Se conserva lo recolectado.")
            break

        if expand_comments:
            expandir_comentarios(driver, panel_scroll, empresa, botones_mas_intentados)
        resenas_visibles = extraer_resenas(panel_scroll, empresa, max_reviews)

        nuevas = 0
        for resena in resenas_visibles:
            clave = (resena["comentario"], resena["empresa"], resena["sede"])
            if clave not in resenas_por_clave:
                resenas_por_clave[clave] = resena
                nuevas += 1

        logging.info(
            "Scroll %s/%s | resenas acumuladas: %s | nuevas: %s",
            intento + 1,
            SCROLL_ATTEMPTS,
            len(resenas_por_clave),
            nuevas,
        )

        if len(resenas_por_clave) >= max_reviews:
            break

        if nuevas == 0:
            intentos_sin_nuevas += 1
        else:
            intentos_sin_nuevas = 0

        if intentos_sin_nuevas >= 6:
            break

        panel_scroll = hacer_scroll_resenas(driver, panel_scroll, empresa)
        dormir(DELAY_SCROLL_MIN, DELAY_SCROLL_MAX)
        cerrar_modal_compartir(driver)
        panel_actual = recuperar_panel_empresa(driver, empresa)
        if panel_actual is not None:
            panel_scroll = panel_actual

    return list(resenas_por_clave.values())[:max_reviews]


def expandir_comentarios(driver, panel_scroll, empresa, botones_mas_intentados):
    botones_mas = panel_scroll.find_elements(
        By.XPATH,
        './/div[contains(@class, "jftiEf")]//button['
        '(normalize-space(.)="Más" or normalize-space(.)="More") and '
        'not(contains(@aria-label, "Compartir")) and not(contains(@aria-label, "Share"))]',
    )
    logging.info("Botones 'Mas' encontrados: %s", len(botones_mas))

    for boton in botones_mas:
        try:
            if not boton.is_displayed() or normalizar_texto(boton.text) not in ["Más", "More"]:
                continue

            tarjeta = boton.find_element(By.XPATH, './ancestor::div[contains(@class, "jftiEf")][1]')
            clave_boton = normalizar_texto(tarjeta.text)[:300]
            if clave_boton in botones_mas_intentados:
                continue

            botones_mas_intentados.add(clave_boton)

            url_previa = driver.current_url
            driver.execute_script("arguments[0].click();", boton)
            time.sleep(0.2)

            if cerrar_modal_compartir(driver):
                logging.warning("Se cerro modal de compartir abierto al expandir un comentario.")
                continue

            if esta_en_reviews_de_usuario(driver):
                cerrar_pestana_reviews_usuario(driver)
                recuperar_panel_empresa(driver, empresa)
                continue

            if driver.current_url != url_previa and "/maps/place/" not in driver.current_url:
                driver.back()
                time.sleep(1)
                recuperar_panel_empresa(driver, empresa)
        except SELENIUM_RECUPERABLE_ERRORS:
            continue


def extraer_resenas(panel_scroll, empresa, max_reviews):
    tarjetas = panel_scroll.find_elements(By.XPATH, './/div[contains(@class, "jftiEf")]')[:max_reviews]
    logging.info("Tarjetas a procesar: %s", len(tarjetas))

    resenas = []

    for tarjeta in tarjetas:
        try:
            comentario = tarjeta.find_element(By.CLASS_NAME, "wiI7pd").text
        except SELENIUM_RECUPERABLE_ERRORS:
            continue

        comentario = normalizar_texto(comentario)
        if not comentario:
            continue

        estrellas = extraer_estrellas(tarjeta)
        if estrellas is None:
            logging.warning("Resena omitida porque no se pudo extraer estrellas.")
            continue

        resenas.append(
            {
                "comentario": comentario,
                "empresa": empresa["nombre"],
                "sede": empresa["sede"],
                "rubro": empresa["rubro"],
                "estrellas": estrellas,
                "sentimiento_estrella": sentimiento_por_estrellas(estrellas),
                "fecha_resena": extraer_fecha(tarjeta),
                "url": empresa["url"],
            }
        )

    return resenas


def scrape_google_maps(driver, empresa, max_reviews=MAX_REVIEWS_PER_COMPANY, expand_comments=True):
    try:
        wait = WebDriverWait(driver, 20)

        logging.info("Abriendo %s - %s", empresa["nombre"], empresa["sede"])
        driver.get(empresa["url"])
        time.sleep(10)

        aceptar_cookies(driver, wait)

        if not abrir_panel_resenas(driver, wait):
            logging.warning("No se encontro el boton de resenas para %s.", empresa["nombre"])
            return []

        panel_scroll = encontrar_panel_scroll(driver)
        if panel_scroll is None:
            logging.warning("No se encontro el panel de resenas para %s.", empresa["nombre"])
            return []

        return recolectar_resenas_con_scroll(
            driver,
            panel_scroll,
            empresa,
            max_reviews,
            expand_comments=expand_comments,
        )

    except SELENIUM_RECUPERABLE_ERRORS as error:
        logging.error("Error de Selenium con %s: %s", empresa["nombre"], error)
        return []


def guardar_dataset(registros, output_file=OUTPUT_FILE):
    if not registros:
        return pd.DataFrame()

    nuevo_df = pd.DataFrame(registros)

    if output_file.exists():
        anterior_df = pd.read_csv(output_file)
        df = pd.concat([anterior_df, nuevo_df], ignore_index=True)
    else:
        df = nuevo_df

    df = df.drop_duplicates(subset=["comentario", "empresa", "sede"])
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    return df


def contar_resenas_existentes(output_file):
    if not output_file.exists():
        return {}

    df = pd.read_csv(output_file).fillna("")
    columnas_requeridas = {"comentario", "empresa", "sede"}
    if not columnas_requeridas.issubset(df.columns):
        logging.warning("El CSV existente no tiene columnas suficientes para omitir sedes completas.")
        return {}

    df = df.drop_duplicates(subset=["comentario", "empresa", "sede"])
    for columna in ["empresa", "sede"]:
        df[columna] = df[columna].astype(str).map(normalizar_texto)

    return df.groupby(["empresa", "sede"]).size().to_dict()


def obtener_argumentos():
    parser = argparse.ArgumentParser(
        description="Scraper de resenas de Google Maps para empresas peruanas."
    )
    parser.add_argument(
        "--empresas",
        type=Path,
        default=EMPRESAS_FILE,
        help="Archivo CSV con columnas nombre,sede,rubro,url.",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=MAX_REVIEWS_PER_COMPANY,
        help="Cantidad maxima de resenas por empresa.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Archivo CSV de salida.",
    )
    parser.add_argument(
        "--limit-companies",
        type=int,
        default=None,
        help="Limita la cantidad de sedes a procesar. Util para pruebas.",
    )
    parser.add_argument(
        "--expand-comments",
        action="store_true",
        default=True,
        help="Hace click en 'Mas' para expandir comentarios. Esta activado por defecto.",
    )
    parser.add_argument(
        "--no-expand-comments",
        action="store_false",
        dest="expand_comments",
        help="No hace click en 'Mas'. Util si Google Maps abre reseñas individuales.",
    )
    parser.add_argument(
        "--pausa-sede-min",
        type=float,
        default=PAUSA_SEDE_MIN,
        help="Pausa minima en segundos entre sedes (con jitter). Sube para ir mas lento y seguro.",
    )
    parser.add_argument(
        "--pausa-sede-max",
        type=float,
        default=PAUSA_SEDE_MAX,
        help="Pausa maxima en segundos entre sedes (con jitter).",
    )
    parser.add_argument(
        "--max-bloqueos",
        type=int,
        default=MAX_BLOQUEOS_CONSECUTIVOS,
        help="Detiene el scraping tras este numero de bloqueos consecutivos de Google.",
    )
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    empresas = cargar_empresas(args.empresas)
    if args.limit_companies is not None:
        empresas = empresas[: args.limit_companies]

    resenas_existentes = contar_resenas_existentes(args.output)
    empresas_pendientes = []
    for empresa in empresas:
        clave = (empresa["nombre"], empresa["sede"])
        total_existente = resenas_existentes.get(clave, 0)
        if total_existente >= args.max_reviews:
            logging.info(
                "Omitiendo %s - %s: ya tiene %s/%s resenas.",
                empresa["nombre"],
                empresa["sede"],
                total_existente,
                args.max_reviews,
            )
            continue

        empresas_pendientes.append(empresa)

    logging.info("Archivo de empresas: %s", args.empresas)
    logging.info("Empresas validas cargadas: %s", len(empresas))
    logging.info("Empresas pendientes por scrapear: %s", len(empresas_pendientes))
    logging.info("Inicio del scraping. Archivo de salida: %s", args.output)
    logging.info("Maximo de resenas por empresa: %s", args.max_reviews)
    logging.info("Expandir comentarios con clicks: %s", "si" if args.expand_comments else "no")

    if empresas_pendientes:
        driver = crear_driver()

        bloqueos_consecutivos = 0
        try:
            for empresa in empresas_pendientes:
                logging.info("Scrapeando %s...", empresa["nombre"])
                resenas = scrape_google_maps(
                    driver,
                    empresa,
                    max_reviews=args.max_reviews,
                    expand_comments=args.expand_comments,
                )
                df = guardar_dataset(resenas, output_file=args.output)

                logging.info("Resenas nuevas obtenidas para %s: %s", empresa["nombre"], len(resenas))
                if not df.empty:
                    logging.info("Total acumulado en CSV: %s", len(df))

                # Si Google muestra una pagina de verificacion, se frena con backoff en vez de insistir.
                bloqueos_consecutivos = esperar_si_bloqueado(driver, bloqueos_consecutivos)
                if bloqueos_consecutivos >= args.max_bloqueos:
                    logging.error(
                        "Demasiados bloqueos consecutivos (%s). Se detiene el scraping y se conserva lo recolectado.",
                        bloqueos_consecutivos,
                    )
                    break

                # Pausa humana (con jitter) entre sedes para no parecer un bot.
                dormir(args.pausa_sede_min, args.pausa_sede_max)
        except KeyboardInterrupt:
            logging.warning("Scraping interrumpido por el usuario. Se conserva el CSV acumulado.")
        finally:
            try:
                driver.quit()
            except WebDriverException:
                pass
    else:
        logging.info("Todas las sedes ya tienen al menos %s resenas.", args.max_reviews)

    if args.output.exists():
        df = pd.read_csv(args.output)
        logging.info("CSV generado correctamente: %s", args.output)
        logging.info("Resumen de sentimientos:\n%s", df["sentimiento_estrella"].value_counts())
    else:
        logging.warning("No se obtuvieron resenas.")


if __name__ == "__main__":
    main()
