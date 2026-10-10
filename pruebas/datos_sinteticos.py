# -*- coding: utf-8 -*-
"""Genera microdatos SINTÉTICOS con la estructura de las tablas de la ENDI R2 que usa el notebook.

Sirven solo para comprobar que el código corre de principio a fin; los valores no tienen ningún significado.
También genera una tabla LMS aproximada (no es la tabla de la OMS).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat


def generar(carpeta, n_ninos=3000, n_upm=300, semilla=1):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)

    # tabla LMS aproximada; el mes 24 aparece dos veces (longitud y talla), como en las tablas de la OMS
    filas = []
    for sexo, desplazamiento in [('M', 0.8), ('F', 0.0)]:
        for mes in range(61):
            mediana = 49.5 + desplazamiento + 26 * (1 - np.exp(-mes / 9)) + 0.33 * mes
            filas.append((sexo, mes, 1.0, mediana, 0.036))
            if mes == 24:
                filas.append((sexo, mes, 1.0, mediana - 0.7, 0.036))
    tabla = pd.DataFrame(filas, columns=['sexo', 'mes', 'L', 'M', 'S'])
    tabla.to_csv(carpeta / 'lms_talla_edad_oms.csv', index=False)
    lms = tabla.drop_duplicates(['sexo', 'mes'], keep='last').set_index(['sexo', 'mes'])

    def talla_de_z(z, sexo, edad):
        L, M, S = lms.loc[(sexo, int(round(min(edad, 60))))]
        return M * (1 + L * S * z) ** (1 / L)

    # niños
    ninos = pd.DataFrame({'id_upm': [f'{u:05d}' for u in rng.integers(0, n_upm, n_ninos)],
                          'id_viv': [f'{v:02d}' for v in rng.integers(1, 20, n_ninos)]})
    ninos['id_hogar'] = '1'
    ninos['id_per'] = [f'{i:04d}' for i in range(n_ninos)]
    ninos = ninos.drop_duplicates(['id_upm', 'id_viv', 'id_hogar', 'id_per']).reset_index(drop=True)
    n = len(ninos)
    ninos['id_mef'] = ''
    ninos['fexp'] = rng.uniform(20, 400, n)
    ninos['estrato'] = rng.integers(1, 30, n).astype(float)
    ninos['prov'] = rng.integers(1, 25, n).astype(float)
    ninos['area'] = rng.integers(1, 3, n).astype(float)
    ninos['etnia'] = rng.choice([1, 2, 3, 4, 5], n, p=[.12, .03, .02, .82, .01]).astype(float)
    edad_m = rng.uniform(0, 59.9, n)
    ninos['edaddias'] = np.round(edad_m * 30.4375)
    ninos['f1_s1_2'] = rng.integers(1, 3, n).astype(float)
    z_real = rng.normal(-0.9, 1.1, n)
    ninos['dcronica'] = (z_real < -2).astype(float)
    ninos['dcronica2_5'] = np.where(edad_m >= 24, (z_real < -2).astype(float), np.nan)
    ninos['nivins_mef'] = np.nan
    sexo = np.where(ninos.f1_s1_2 == 1, 'M', 'F')
    medida = np.array([talla_de_z(z + rng.normal(0, 0.1), s, e) for z, s, e in zip(z_real, sexo, edad_m)])
    for j in (1, 2, 3):
        ninos[f'f1_s5_5_{j}'] = np.where(edad_m < 24, medida + rng.normal(0, 0.2, n), np.nan)
        ninos[f'f1_s5_6_{j}'] = np.where(edad_m >= 24, medida + rng.normal(0, 0.2, n), np.nan)

    # madres: otras filas del mismo hogar
    madres = ninos[['id_upm', 'id_viv', 'id_hogar']].copy()
    madres['id_per'] = [f'M{i:04d}' for i in range(n)]
    madres['id_mef'] = [f'{i:04d}' for i in range(n)]
    madres['nivins_mef'] = rng.choice([1, 2, 3, np.nan], n, p=[.3, .4, .26, .04])
    for c in ninos.columns:
        if c not in madres.columns:
            madres[c] = '' if ninos[c].dtype == object else np.nan
    personas = pd.concat([ninos, madres[ninos.columns]], ignore_index=True)
    for c in ['id_upm', 'id_viv', 'id_hogar', 'id_per', 'id_mef']:
        personas[c] = personas[c].astype(str).replace('nan', '')
    pyreadstat.write_dta(personas, str(carpeta / 'BDD_ENDI_R2_f1_personas.dta'))

    # f2_salud_ninez: vínculo con la madre, nacimiento y carné (hasta 12 mediciones)
    sn = ninos[['id_upm', 'id_viv', 'id_hogar', 'id_per']].copy()
    sn['id_mef'] = madres['id_mef'].values
    sn['f2_s4d_432'] = rng.choice([np.nan, *range(30, 42)], n)
    sn['f2_s4d_443_b'] = np.where(rng.random(n) < .2, np.nan, rng.normal(3100, 450, n))
    sn['f2_s4d_444'] = np.nan
    for c in ['f2_s4d_433', 'f2_s4d_446', 'f2_s4f_462', 'f2_s4f_463_a', 'f2_s4f_463_b']:
        sn[c] = np.nan          # columnas de la base real que el notebook no usa
    carne = {}
    for i in range(1, 13):
        for p in ('edad', 'peso', 'talla'):
            carne[f'f2_s4f_464_{p}_{i}'] = np.full(n, np.nan)
    for r in range(n):
        k = rng.integers(0, 13)
        if k == 0:
            continue
        edades = np.sort(rng.uniform(0, max(edad_m[r], 0.5), k)).round()      # meses enteros: hay empates
        for i, e in enumerate(edades, start=1):
            carne[f'f2_s4f_464_edad_{i}'][r] = e
            carne[f'f2_s4f_464_talla_{i}'][r] = talla_de_z(z_real[r] + 0.45 + rng.normal(0, 0.7), sexo[r], e)
            carne[f'f2_s4f_464_peso_{i}'][r] = 3 + e * 0.25
    sn = pd.concat([sn, pd.DataFrame(carne)], axis=1)
    pyreadstat.write_dta(sn, str(carpeta / 'BDD_ENDI_R2_f2_salud_ninez.dta'))
    return carpeta


if __name__ == '__main__':
    import sys
    destino = generar(sys.argv[1] if len(sys.argv) > 1 else 'datos_sinteticos')
    print('datos sintéticos en', Path(destino).resolve())
