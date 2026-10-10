# -*- coding: utf-8 -*-
"""Construye datos/lms_talla_edad_oms.csv a partir de las tablas mensuales de la OMS.

Fuente: WHO Child Growth Standards, longitud/talla para la edad
https://www.who.int/tools/child-growth-standards/standards/length-height-for-age

Descargue las cuatro tablas de puntajes z (z-scores) mensuales, en Excel o texto:
  - niños (boys), longitud para la edad, 0 a 2 años
  - niños (boys), talla para la edad, 2 a 5 años
  - niñas (girls), longitud para la edad, 0 a 2 años
  - niñas (girls), talla para la edad, 2 a 5 años
Cada tabla tiene una columna de edad en meses (Month) y los parámetros L, M y S.

Uso:
    python scripts/construir_lms_oms.py --ninos-0-2 A --ninos-2-5 B --ninas-0-2 C --ninas-2-5 D

La salida tiene las columnas sexo (M/F), mes, L, M y S. El mes 24 aparece dos veces, primero con longitud y
luego con talla; el notebook conserva la última fila, es decir, usa la talla desde los 24 meses.
Para leer archivos .xlsx se necesita openpyxl (pip install openpyxl).
"""
import argparse
from pathlib import Path

import pandas as pd


def leer_tabla(ruta):
    ruta = Path(ruta)
    if ruta.suffix.lower() in ('.xlsx', '.xls'):
        df = pd.read_excel(ruta)
    else:
        df = pd.read_csv(ruta, sep=None, engine='python')
    columnas = {c.strip().lower(): c for c in df.columns.astype(str)}
    edad = next((columnas[c] for c in ('month', 'months', 'mes', 'meses') if c in columnas), None)
    if edad is None or not {'l', 'm', 's'} <= set(columnas):
        raise ValueError(f'{ruta.name}: se esperaban las columnas Month, L, M y S; hay {list(df.columns)}')
    t = df[[edad, columnas['l'], columnas['m'], columnas['s']]].copy()
    t.columns = ['mes', 'L', 'M', 'S']
    t = t.dropna()
    t['mes'] = t.mes.astype(int)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ninos-0-2', required=True)
    ap.add_argument('--ninos-2-5', required=True)
    ap.add_argument('--ninas-0-2', required=True)
    ap.add_argument('--ninas-2-5', required=True)
    ap.add_argument('--salida', default=str(Path(__file__).resolve().parents[1] / 'datos' / 'lms_talla_edad_oms.csv'))
    a = ap.parse_args()

    partes = []
    for sexo, t02, t25 in [('M', a.ninos_0_2, a.ninos_2_5), ('F', a.ninas_0_2, a.ninas_2_5)]:
        corta, larga = leer_tabla(t02), leer_tabla(t25)
        if corta.mes.min() != 0 or corta.mes.max() != 24 or larga.mes.min() != 24 or larga.mes.max() != 60:
            raise ValueError(f'rangos de meses inesperados para sexo {sexo}: '
                             f'{corta.mes.min()}-{corta.mes.max()} y {larga.mes.min()}-{larga.mes.max()}')
        partes += [corta.assign(sexo=sexo), larga.assign(sexo=sexo)]      # primero longitud, luego talla

    lms = pd.concat(partes, ignore_index=True)[['sexo', 'mes', 'L', 'M', 'S']]
    salida = Path(a.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    lms.to_csv(salida, index=False)
    final = lms.sort_values(['sexo', 'mes']).drop_duplicates(['sexo', 'mes'], keep='last')
    print(f'{salida}: {len(lms)} filas ({len(final)} tras conservar la talla en el mes 24)')


if __name__ == '__main__':
    main()
