import argparse
import logging
import re
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


DEBUGGER_ADDRESS = "127.0.0.1:9222"
EMPRESAS_FILE = Path("empresas.csv")
OUTPUT_FILE = Path("dataset_consumidores_peru.csv")
MAX_REVIEWS_PER_COMPANY = 40
SCROLL_ATTEMPTS = 25
SCROLL_PAUSE_SECONDS = 2
COLUMNAS_EMPRESAS = ["nombre", "sede", "rubro", "url"]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def crear_driver():
    options = Options()
    options.debugger_address = DEBUGGER_ADDRESS
    return webdriver.Chrome(options=options)


def normalizar_texto(texto):
    return re.sub(r"\s+", " ", texto).strip()


def cargar_empresas(empresas_file):
    if not empresas_file.exists():
        raise FileNotFoundError(f"No existe el archivo de empresas: {empresas_file}")

    df = pd.read_csv(empresas_file)
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
    df = df.drop_duplicates(subset=["nombre", "sede", "url"])

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
    except WebDriverException:
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
        except WebDriverException:
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

    for xpath in posibles_scroll:
        elementos = driver.find_elements(By.XPATH, xpath)
        for elemento in elementos:
            try:
                if elemento.is_displayed() and elemento.size["height"] > 0:
                    return elemento
            except WebDriverException:
                continue

    return None


def contar_tarjetas(driver):
    return len(driver.find_elements(By.XPATH, '//div[contains(@class, "jftiEf")]'))


def cargar_resenas(driver, panel_scroll, max_reviews):
    tarjetas_previas = 0
    intentos_sin_cambios = 0

    for intento in range(SCROLL_ATTEMPTS):
        driver.execute_script(
            "arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].clientHeight;",
            panel_scroll,
        )
        time.sleep(SCROLL_PAUSE_SECONDS)

        tarjetas_actuales = contar_tarjetas(driver)
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


def hacer_scroll_resenas(driver, panel_scroll):
    driver.execute_script(
        "arguments[0].scrollTop = arguments[0].scrollTop + 1500;",
        panel_scroll,
    )
    driver.execute_script(
        "arguments[0].dispatchEvent(new WheelEvent('wheel', "
        "{deltaY: 1500, bubbles: true, cancelable: true}));",
        panel_scroll,
    )

    tarjetas = driver.find_elements(By.XPATH, '//div[contains(@class, "jftiEf")]')
    if tarjetas:
        driver.execute_script("arguments[0].scrollIntoView({block: 'end'});", tarjetas[-1])

    try:
        panel_scroll.click()
        panel_scroll.send_keys(Keys.PAGE_DOWN)
    except WebDriverException:
        try:
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.PAGE_DOWN)
        except WebDriverException:
            pass


def recolectar_resenas_con_scroll(driver, panel_scroll, empresa, max_reviews):
    resenas_por_clave = {}
    intentos_sin_nuevas = 0

    for intento in range(SCROLL_ATTEMPTS):
        expandir_comentarios(driver)
        resenas_visibles = extraer_resenas(driver, empresa, max_reviews)

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

        hacer_scroll_resenas(driver, panel_scroll)
        time.sleep(SCROLL_PAUSE_SECONDS)

    return list(resenas_por_clave.values())[:max_reviews]


def expandir_comentarios(driver):
    botones_mas = driver.find_elements(By.XPATH, '//button[contains(., "Más")]')
    logging.info("Botones 'Mas' encontrados: %s", len(botones_mas))

    for boton in botones_mas:
        try:
            boton.click()
            time.sleep(0.2)
        except WebDriverException:
            continue


def extraer_resenas(driver, empresa, max_reviews):
    tarjetas = driver.find_elements(By.XPATH, '//div[contains(@class, "jftiEf")]')[:max_reviews]
    logging.info("Tarjetas a procesar: %s", len(tarjetas))

    resenas = []

    for tarjeta in tarjetas:
        try:
            comentario = tarjeta.find_element(By.CLASS_NAME, "wiI7pd").text
        except WebDriverException:
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


def scrape_google_maps(driver, empresa, max_reviews=MAX_REVIEWS_PER_COMPANY):
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

        return recolectar_resenas_con_scroll(driver, panel_scroll, empresa, max_reviews)

    except WebDriverException as error:
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
    return parser.parse_args()


def main():
    args = obtener_argumentos()
    empresas = cargar_empresas(args.empresas)
    if args.limit_companies is not None:
        empresas = empresas[: args.limit_companies]

    logging.info("Archivo de empresas: %s", args.empresas)
    logging.info("Empresas validas cargadas: %s", len(empresas))
    logging.info("Inicio del scraping. Archivo de salida: %s", args.output)
    logging.info("Maximo de resenas por empresa: %s", args.max_reviews)

    driver = crear_driver()

    try:
        for empresa in empresas:
            logging.info("Scrapeando %s...", empresa["nombre"])
            resenas = scrape_google_maps(driver, empresa, max_reviews=args.max_reviews)
            df = guardar_dataset(resenas, output_file=args.output)

            logging.info("Resenas nuevas obtenidas para %s: %s", empresa["nombre"], len(resenas))
            if not df.empty:
                logging.info("Total acumulado en CSV: %s", len(df))

            time.sleep(3)
    except KeyboardInterrupt:
        logging.warning("Scraping interrumpido por el usuario. Se conserva el CSV acumulado.")
    finally:
        try:
            driver.quit()
        except WebDriverException:
            pass

    if args.output.exists():
        df = pd.read_csv(args.output)
        logging.info("CSV generado correctamente: %s", args.output)
        logging.info("Resumen de sentimientos:\n%s", df["sentimiento_estrella"].value_counts())
    else:
        logging.warning("No se obtuvieron resenas.")


if __name__ == "__main__":
    main()
