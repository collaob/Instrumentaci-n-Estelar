"""
pipeline_ngc300.py - FIS2421 Tarea 2 - Reducción de NGC 300 (telescopio MAS500)
================================================================================

Pasos, en orden:
  1. Verifica que existan las carpetas y archivos necesarios.
  2. Selecciona los FITS por HEADER (IMAGETYP, FILTER, EXPTIME), no por nombre.
  3. Construye el master dark de 300 s, el pedestal y la máscara de píxeles malos.
  4. Construye un master flat por filtro (B y V), con flats de 0.5 s normalizados.
  5. Reduce cada frame de ciencia: (ciencia - dark) / flat, y corrige píxeles malos.
  6. Alinea los frames de cada filtro por estrellas y los combina por mediana,
     usando solo la zona donde el flat es válido y todos los frames tienen dato.
  7. Alinea V sobre B, recorta la zona válida, resta el cielo y genera las imágenes.
  8. Escribe FITS intermedios y finales, PNG, un log y un resumen para el informe.

Uso (desde la carpeta del proyecto):
    python pipeline_ngc300.py

Estructura esperada:
    <proyecto>/
        NGC_300_B_300.000secs_*.fit   ciencia B (en la raíz)
        NGC_300_V_300.000secs_*.fit   ciencia V (en la raíz)
        darks/                        darks (se usan los de 300 s)
        flats/                        flats B y V (se usan los de 0.5 s)
        BPM.fit                       máscara del observatorio (1 = bueno, 0 = malo)
        pruebas/                      ensayos (este script NO los lee)

Requiere:
    pip install astropy numpy scipy astroalign matplotlib
"""

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import astroalign as aa
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import (AsinhStretch, ImageNormalize,
                                   PercentileInterval, make_lupton_rgb)
from scipy import ndimage


# =====================================================================
# 1. PARÁMETROS  (cualquier cambio se hace solo aquí)
# =====================================================================
ROOT = Path(__file__).resolve().parent     # carpeta del proyecto
PROC = ROOT / "procesado"                  # datos intermedios y finales
VIS = PROC / "visualizacion"               # imágenes PNG

FILTROS = ("B", "V")
T_CIENCIA = 300.0            # s: exposición de la ciencia y de los darks usados
T_FLAT = 0.5                 # s: exposición de los flats usados (los de 1.02 s quedan fuera)
N_DARKS = 5                  # darks de 300 s esperados
N_FLATS = {"B": 10, "V": 6}  # flats de 0.5 s esperados por filtro
N_CIENCIA = 3                # frames de ciencia por filtro esperados
SIGMA_CALIENTES = 10         # umbral de píxeles calientes (en sigma robusto del dark)
MIN_FLAT = 0.2               # píxeles del flat normalizado por debajo de esto = malos
FRAC_ZONA = 0.95             # fracción mínima de píxeles válidos para recortar
PIXSCALE = 0.316             # arcsec/px (21.57' x 60 / 4096 px, presentación MAS500)

PROC.mkdir(exist_ok=True)
VIS.mkdir(exist_ok=True)


# =====================================================================
# 2. REGISTRO (consola + archivo para citar en el informe)
# =====================================================================
log = logging.getLogger("pipeline_ngc300")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(message)s")
for _handler in (logging.FileHandler(PROC / "log_reduccion.txt", mode="w", encoding="utf-8"),
                 logging.StreamHandler(sys.stdout)):
    _handler.setFormatter(_fmt)
    log.addHandler(_handler)


# =====================================================================
# 3. FUNCIONES AUXILIARES
# =====================================================================
def leer(ruta):
    """Devuelve (datos float32, header). memmap=False evita el error con BZERO."""
    datos, cab = fits.getdata(ruta, header=True, memmap=False)
    return datos.astype(np.float32), cab


def buscar(carpeta, tipo, exptime=None, filtro=None):
    """
    Lista [(ruta, header)] de los FITS que están DIRECTAMENTE dentro de 'carpeta'
    y cumplen:
      tipo:    texto que debe aparecer en IMAGETYP ('dark', 'flat' o 'light')
      exptime: EXPTIME en segundos (None = cualquiera)
      filtro:  FILTER 'B' o 'V' (None = cualquiera)
    No entra a subcarpetas, así los ensayos de 'pruebas/' nunca se mezclan.
    """
    encontrados = []
    for ruta in sorted(Path(carpeta).glob("*.fit*")):
        cab = fits.getheader(ruta, memmap=False)
        if tipo not in str(cab.get("IMAGETYP", "")).lower():
            continue
        if exptime is not None and abs(float(cab.get("EXPTIME", -1)) - exptime) > 0.01:
            continue
        if filtro is not None and str(cab.get("FILTER", "")).strip() != filtro:
            continue
        encontrados.append((ruta, cab))
    return encontrados


def aviso_conteo(lista, esperados, nombre):
    """Avisa en el log si el número de archivos no es el esperado."""
    if len(lista) != esperados:
        log.warning(f"AVISO: {nombre}: encontrados {len(lista)}, esperados {esperados}")


def verificar_entrada():
    """Revisa que estén las piezas necesarias. Si falta algo grave, detiene el programa."""
    problemas = []
    for carpeta in ("darks", "flats"):
        if not (ROOT / carpeta).is_dir():
            problemas.append(f"no existe la carpeta '{carpeta}/'")
    if not (ROOT / "BPM.fit").exists():
        problemas.append("falta BPM.fit (renombra BPM__1_.fit)")
    for f in FILTROS:
        if not buscar(ROOT, "light", exptime=T_CIENCIA, filtro=f):
            problemas.append(f"no hay ciencia {f} de {T_CIENCIA:.0f} s en la raíz")
    if problemas:
        for p in problemas:
            log.error("ERROR: " + p)
        sys.exit(1)


# =====================================================================
# 4. MASTER DARK, PEDESTAL Y MÁSCARA DE PÍXELES MALOS
# =====================================================================
def construir_dark():
    """
    Combina los darks de 300 s por mediana.
    El pedestal (nivel medio del dark) hace de bias, porque no tenemos bias.
    Los píxeles malos son los del observatorio (BPM) más los calientes.
    """
    darks = buscar(ROOT / "darks", "dark", exptime=T_CIENCIA)
    aviso_conteo(darks, N_DARKS, f"darks {T_CIENCIA:.0f} s")
    if not darks:
        sys.exit("No hay darks de 300 s en darks/")

    master = np.median(np.stack([leer(r)[0] for r, _ in darks]), axis=0)
    pedestal = float(np.median(master))

    # Píxeles calientes: exceso sobre el pedestal, en sigma robusto (MAD)
    exceso = master - pedestal
    sigma = 1.4826 * np.median(np.abs(exceso - np.median(exceso)))
    calientes = exceso > SIGMA_CALIENTES * sigma

    bpm = fits.getdata(ROOT / "BPM.fit", memmap=False).astype(bool)  # True = bueno
    malos = (~bpm) | calientes

    log.info(f"Dark: {len(darks)} frames de {T_CIENCIA:.0f} s | pedestal {pedestal:.1f} ADU")
    log.info(f"Píxeles calientes: {calientes.sum()} ({calientes.mean():.3%})")
    log.info(f"Píxeles malos del BPM: {(~bpm).sum()} ({(~bpm).mean():.3%})")

    fits.writeto(PROC / "master_dark_300s.fits", master, overwrite=True)
    fits.writeto(PROC / "mascara_malos.fits", malos.astype(np.uint8), overwrite=True)

    return master, pedestal, malos, len(darks), int(calientes.sum()), int((~bpm).sum())


# =====================================================================
# 5. MASTER FLAT POR FILTRO
# =====================================================================
def construir_flat(filtro, pedestal):
    """
    Combina los flats de 0.5 s de un filtro:
    resta el pedestal, normaliza cada flat por su mediana y toma la mediana de todos.
    Devuelve el master flat y la máscara 'malo_flat' (True donde el flat no es confiable,
    por ejemplo en las esquinas del campo). Esa máscara define la zona válida.
    """
    flats = buscar(ROOT / "flats", "flat", exptime=T_FLAT, filtro=filtro)
    aviso_conteo(flats, N_FLATS[filtro], f"flats {filtro} {T_FLAT} s")
    if not flats:
        sys.exit(f"No hay flats {filtro} de {T_FLAT} s en flats/")

    normalizados = []
    for ruta, _ in flats:
        d = leer(ruta)[0] - pedestal
        normalizados.append(d / np.median(d))

    mf = np.median(np.stack(normalizados), axis=0)
    mf /= np.nanmedian(mf)
    malo = ~np.isfinite(mf) | (mf < MIN_FLAT)

    # Donde el flat no es confiable se usa 1.0 para no dividir por cero,
    # pero esos píxeles quedan FUERA de la zona válida (malo_flat).
    mf_seguro = np.where(malo, 1.0, mf).astype(np.float32)

    log.info(f"Flat {filtro}: {len(flats)} frames | píxeles fuera de zona por flat: {malo.sum()}")
    fits.writeto(PROC / f"master_flat_{filtro}.fits", mf_seguro, overwrite=True)
    return mf_seguro, malo


# =====================================================================
# 6. REDUCCIÓN DE LA CIENCIA
# =====================================================================
def reducir(filtro, dark, mf, malos_base):
    """
    Para cada frame de ciencia del filtro:
      dk = ciencia - dark          (quita corriente térmica y pedestal)
      fl = dk / flat               (quita la respuesta no uniforme)
      píxeles malos (BPM y calientes) -> mediana de vecinos
    Devuelve la lista de frames reducidos y el header del primero.
    """
    ciencia = buscar(ROOT, "light", exptime=T_CIENCIA, filtro=filtro)
    aviso_conteo(ciencia, N_CIENCIA, f"ciencia {filtro}")
    if not ciencia:
        sys.exit(f"No hay ciencia {filtro} en la raíz")

    frames = []
    cab_ref = ciencia[0][1]

    for i, (ruta, _) in enumerate(ciencia):
        raw, _ = leer(ruta)
        dk = raw - dark
        fl = dk / mf

        if i == 0:  # intermedios del primer frame, para mostrar el proceso en el informe
            fits.writeto(PROC / f"int_{filtro}_1_cruda.fits", raw, overwrite=True)
            fits.writeto(PROC / f"int_{filtro}_2_dark.fits", dk, overwrite=True)
            fits.writeto(PROC / f"int_{filtro}_3_flat.fits", fl, overwrite=True)

        fl[malos_base] = ndimage.median_filter(fl, size=3)[malos_base]
        frames.append(fl)
        log.info(f"  reducido {ruta.name}")

    return frames, cab_ref


# =====================================================================
# 7. ALINEACIÓN Y COMBINACIÓN
# =====================================================================
def cobertura_de(T, ref):
    """
    Devuelve True donde la transformación T lleva datos reales al marco de 'ref'.
    Se transforma una imagen de unos: donde sale cerca de 1 hay datos,
    donde sale 0 no los hay. No usa la huella de astroalign, que en esta
    versión devuelve todo False.
    """
    unos = np.ones(ref.shape, dtype=np.float32)
    cob, _ = aa.apply_transform(T, unos, ref, fill_value=0.0)
    return cob > 0.5


def alinear_y_combinar(frames, zona_inicial):
    """
    Alinea cada frame al primero y los combina por mediana.

    zona_inicial: píxeles válidos según el flat (True = confiable).
    La zona final es la intersección de esa zona con la cobertura de
    todos los frames alineados. Fuera de la zona el resultado es NaN.
    """
    ref = frames[0].astype(np.float32)
    usados = [ref]
    zona = zona_inicial.copy()

    for k, fr in enumerate(frames[1:], start=2):
        try:
            T, _ = aa.find_transform(fr, ref)
        except Exception as e:
            log.warning(f"  AVISO: el frame {k} no se pudo alinear ({e}); se descarta")
            continue
        alineado, _ = aa.apply_transform(T, fr, ref, fill_value=0.0)
        usados.append(alineado.astype(np.float32))
        zona &= cobertura_de(T, ref)

    fraccion = zona.mean()
    log.info(f"  zona común: {fraccion:.1%} de la imagen")
    if fraccion < 0.5:
        sys.exit("La zona común es menor al 50 %: revisa la alineación antes de seguir")

    pila = [np.where(zona, a, np.nan).astype(np.float32) for a in usados]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # NaN fuera de zona: esperado
        combinada = np.nanmedian(np.stack(pila), axis=0)
    return combinada, zona, len(usados)


def alinear_V_a_B(V, zona_V, B):
    """
    Transforma V para que coincida con B (necesario para el color).
    La zona final es donde la transformación tiene datos reales en B
    y donde V tenía dato real antes de transformar.
    """
    V_det = np.where(zona_V, V, 0.0).astype(np.float32)
    B_det = np.nan_to_num(B, nan=0.0).astype(np.float32)

    T, _ = aa.find_transform(V_det, B_det)
    V_al, _ = aa.apply_transform(T, V_det, B_det, fill_value=0.0)
    zona_al, _ = aa.apply_transform(T, zona_V.astype(np.float32), B_det, fill_value=0.0)
    zona = cobertura_de(T, B_det) & (zona_al > 0.99)
    return B_det, V_al.astype(np.float32), zona


# =====================================================================
# 8. IMÁGENES Y RESUMEN
# =====================================================================
def generar_imagenes(B, V, zona):
    """
    Recorta a la zona válida, resta el cielo y genera:
      gris_B.png, gris_V.png: escala asinh fija
      color.png:              R = V, G = (B+V)/2, B = B
    Devuelve (cielo_B, cielo_V, recorte).
    """
    filas = np.where(zona.mean(axis=1) > FRAC_ZONA)[0]
    cols = np.where(zona.mean(axis=0) > FRAC_ZONA)[0]
    y0, y1 = filas.min(), filas.max() + 1
    x0, x1 = cols.min(), cols.max() + 1

    Bc = B[y0:y1, x0:x1]
    Vc = V[y0:y1, x0:x1]
    zc = zona[y0:y1, x0:x1]

    cielo_B = sigma_clipped_stats(Bc, mask=~zc, sigma=3)[1]
    cielo_V = sigma_clipped_stats(Vc, mask=~zc, sigma=3)[1]
    Bs = np.where(zc, np.clip(Bc - cielo_B, 0, None), 0)
    Vs = np.where(zc, np.clip(Vc - cielo_V, 0, None), 0)

    def guardar_gris(img, nombre):
        norm = ImageNormalize(img, interval=PercentileInterval(99.5),
                              stretch=AsinhStretch(a=0.1))
        plt.imsave(VIS / f"{nombre}.png", np.asarray(norm(img)),
                   cmap="gray", origin="lower")

    def canal(img):
        lo, hi = np.percentile(img[zc], [0.5, 99.5])
        return np.clip((img - lo) / (hi - lo), 0, 1)

    guardar_gris(Bs, "gris_B")
    guardar_gris(Vs, "gris_V")

    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = make_lupton_rgb(canal(Vs), canal((Bs + Vs) / 2), canal(Bs),
                              stretch=0.3, Q=8, minimum=0)
    plt.imsave(VIS / "color.png", rgb, origin="lower")

    return cielo_B, cielo_V, (x0, x1, y0, y1)


def escribir_resumen(info):
    """Guarda los parámetros y resultados clave para el informe."""
    x0, x1, y0, y1 = info["recorte"]
    lineas = [
        "Reducción NGC 300 (pipeline_ngc300.py)",
        f"Fecha de ejecución: {datetime.now():%Y-%m-%d %H:%M}",
        f"Ciencia: {T_CIENCIA:.0f} s | Flats: {T_FLAT} s | Darks usados: {info['n_darks']} x {T_CIENCIA:.0f} s",
        f"Pedestal (aprox. de bias): {info['pedestal']:.1f} ADU",
        f"Píxeles calientes: {info['n_calientes']}",
        f"Píxeles malos del BPM: {info['n_bpm']}",
        f"Cielo B: {info['cielo_B']:.1f} ADU | Cielo V: {info['cielo_V']:.1f} ADU",
        f"Recorte: columnas {x0}-{x1}, filas {y0}-{y1}",
        f"Escala de placa (presentación): {PIXSCALE} arcsec/px",
    ]
    (PROC / "resumen_reduccion.txt").write_text("\n".join(lineas), encoding="utf-8")


# =====================================================================
# 9. PROGRAMA PRINCIPAL
# =====================================================================
def main():
    log.info(f"pipeline_ngc300.py - {datetime.now():%Y-%m-%d %H:%M}")
    log.info(f"Carpeta del proyecto: {ROOT}")

    verificar_entrada()
    dark, pedestal, malos_base, n_darks, n_cal, n_bpm = construir_dark()

    finales = {}
    for f in FILTROS:
        log.info(f"== Filtro {f} ==")
        mf, malo_flat = construir_flat(f, pedestal)
        frames, cab = reducir(f, dark, mf, malos_base)
        comb, zona, n = alinear_y_combinar(frames, zona_inicial=~malo_flat)

        hdr = cab.copy()
        for clave in ("BZERO", "BSCALE"):
            hdr.pop(clave, None)
        hdr["HISTORY"] = f"pipeline_ngc300: {n} frames combinados por mediana"
        fits.writeto(PROC / f"final_{f}.fits", comb, header=hdr, overwrite=True)

        finales[f] = (comb, zona)
        log.info(f"  ciencia {f}: {n} frames combinados")

    log.info("== Alineación V sobre B e imágenes ==")
    B, zona_B = finales["B"]
    V, zona_V = finales["V"]
    Bf, V_al, zona = alinear_V_a_B(V, zona_V, B)
    zona = zona & zona_B
    fits.writeto(PROC / "final_V_alineado_a_B.fits", V_al, overwrite=True)

    cielo_B, cielo_V, recorte = generar_imagenes(Bf, V_al, zona)
    log.info(f"Cielo B: {cielo_B:.1f} ADU | Cielo V: {cielo_V:.1f} ADU")

    escribir_resumen({
        "n_darks": n_darks, "pedestal": pedestal,
        "n_calientes": n_cal, "n_bpm": n_bpm,
        "cielo_B": cielo_B, "cielo_V": cielo_V, "recorte": recorte,
    })

    log.info(f"\nListo. Datos en: {PROC}")
    log.info(f"Imágenes en: {VIS}")
    log.info(f"Registro: {PROC / 'log_reduccion.txt'}")


if __name__ == "__main__":
    main()