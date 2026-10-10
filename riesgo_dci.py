# -*- coding: utf-8 -*-
"""Riesgo de desnutrición crónica infantil (DCI) a partir de las mediciones del carné de control.

El modelo estima la probabilidad de que un niño presente DCI (puntaje Z de talla para la edad < -2 DE)
entre los 24 y los 42 meses, a partir de al menos tres mediciones de longitud tomadas hasta los 18 meses.

Uso desde Python:

    from riesgo_dci import ModeloRiesgoDCI
    modelo = ModeloRiesgoDCI.cargar()                      # modelo/parametros_modelo.json
    modelo.probabilidad([(2, 57.5), (6, 67.0), (12, 75.5), (17, 81.0)], sexo='M')

Uso desde la terminal:

    python riesgo_dci.py --sexo M 2:57.5 6:67 12:75.5 17:81

El módulo solo necesita NumPy: el modelo se guarda como un archivo JSON con los parámetros de la
imputación, la estandarización y la regresión logística, sin objetos serializados de scikit-learn.
Las funciones `puntaje_z` y `rasgos_trayectoria` son las mismas que usa el notebook de entrenamiento,
de modo que las variables se construyen igual al entrenar y al predecir.

El resultado es una estimación de riesgo para apoyar la priorización del seguimiento; no reemplaza la
valoración clínica ni la medición estandarizada.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

__all__ = ['RASGOS', 'puntaje_z', 'mes_lms', 'rasgos_trayectoria', 'exportar_parametros', 'ModeloRiesgoDCI']

# Variables de trayectoria, en el orden que espera el modelo
RASGOS = ['z_ult', 'z_medio', 'z_min', 'z_pri', 'delta', 'pend', 'n_med', 'edad_ult', 'bajo_alguna']

EDAD_MAX_PREDICTORES = 18      # meses: solo se usan mediciones tomadas hasta esta edad
MIN_MEDICIONES = 3             # mínimo de mediciones válidas para estimar el riesgo
RANGO_TALLA_CM = (40, 125)     # rango físico plausible
Z_MAX = 6                      # criterio OMS: |Z| > 6 se considera error de medición
RANGO_PENDIENTE_MESES = 3      # la pendiente solo se calcula si la serie abarca al menos 3 meses

RUTA_POR_DEFECTO = Path(__file__).resolve().parent / 'modelo' / 'parametros_modelo.json'


def mes_lms(edad_meses):
    """Mes entero de la tabla LMS (0 a 60) que corresponde a una edad en meses.
    Redondea al entero más cercano (mitades al par, como pandas y NumPy)."""
    return int(min(max(np.round(float(edad_meses)), 0), 60))


def puntaje_z(talla, L, M, S):
    """Puntaje Z por el método LMS: ((talla / M) ** L - 1) / (L * S).
    Acepta escalares o arreglos de NumPy/pandas."""
    return ((talla / M) ** L - 1) / (L * S)


def rasgos_trayectoria(edades, z):
    """Variables de trayectoria de un niño a partir de sus mediciones (edad en meses y puntaje Z corregido).

    Las mediciones se ordenan por edad de forma estable (los empates conservan el orden de entrada).
    Devuelve un diccionario con las variables de `RASGOS` y la edad de la primera medición (`edad_pri`).
    La pendiente (DE por mes) queda como NaN si hay menos de tres mediciones o si la serie abarca menos
    de tres meses; el modelo la imputa con la mediana de la muestra de entrenamiento.
    """
    e = np.asarray(edades, dtype=float)
    z = np.asarray(z, dtype=float)
    orden = np.argsort(e, kind='stable')
    e, z = e[orden], z[orden]
    d = {'n_med': len(z), 'edad_pri': e[0], 'edad_ult': e[-1],
         'z_pri': z[0], 'z_ult': z[-1], 'z_min': z.min(),
         'z_medio': z.mean(), 'delta': z[-1] - z[0]}
    if len(z) >= 3 and np.ptp(e) >= RANGO_PENDIENTE_MESES:
        d['pend'] = np.polyfit(e, z, 1)[0]
    else:
        d['pend'] = np.nan
    d['bajo_alguna'] = float((z < -2).any())
    return d


def exportar_parametros(pipeline, lms, sesgo_carne, ruta, umbral=None, metadatos=None):
    """Guarda en JSON los parámetros del modelo final entrenado en el notebook.

    `pipeline` es un Pipeline de scikit-learn con los pasos
    prep = ColumnTransformer([('n', Pipeline([('i', SimpleImputer), ('s', StandardScaler)]), RASGOS)])
    y clf = LogisticRegression. `lms` es un DataFrame con columnas sexo, mes, L, M y S.
    """
    num = pipeline.named_steps['prep'].named_transformers_['n']
    imputador, escalador = num.named_steps['i'], num.named_steps['s']
    clf = pipeline.named_steps['clf']
    columnas = list(pipeline.named_steps['prep'].transformers_[0][2])
    if columnas != RASGOS:
        raise ValueError(f'el modelo usa {columnas}; se esperaba {RASGOS}')
    params = {
        'descripcion': 'Regresión logística para la probabilidad de DCI entre los 24 y 42 meses a partir de '
                       'mediciones del carné de control hasta los 18 meses (ENDI R2, Ecuador).',
        'rasgos': RASGOS,
        'mediana_imputacion': [float(v) for v in imputador.statistics_],
        'media_estandarizacion': [float(v) for v in escalador.mean_],
        'escala_estandarizacion': [float(v) for v in escalador.scale_],
        'coeficientes': [float(v) for v in clf.coef_[0]],
        'intercepto': float(clf.intercept_[0]),
        'sesgo_carne_de': float(sesgo_carne),
        'umbral': None if umbral is None else float(umbral),
        'edad_max_meses': EDAD_MAX_PREDICTORES,
        'min_mediciones': MIN_MEDICIONES,
        'lms': {sexo: {str(int(r.mes)): [float(r.L), float(r.M), float(r.S)] for r in g.itertuples()}
                for sexo, g in lms.groupby('sexo')},
        'metadatos': metadatos or {},
    }
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding='utf-8')
    return params


class ModeloRiesgoDCI:
    """Modelo de riesgo de DCI cargado desde el archivo JSON de parámetros."""

    def __init__(self, params):
        self.params = params
        self.rasgos = list(params['rasgos'])
        self._mediana = np.asarray(params['mediana_imputacion'], dtype=float)
        self._media = np.asarray(params['media_estandarizacion'], dtype=float)
        self._escala = np.asarray(params['escala_estandarizacion'], dtype=float)
        self._coef = np.asarray(params['coeficientes'], dtype=float)
        self._intercepto = float(params['intercepto'])
        self.sesgo = float(params['sesgo_carne_de'])
        self.umbral = params.get('umbral')
        self._lms = {s: {int(m): v for m, v in t.items()} for s, t in params['lms'].items()}

    @classmethod
    def cargar(cls, ruta=None):
        ruta = Path(ruta) if ruta is not None else RUTA_POR_DEFECTO
        return cls(json.loads(ruta.read_text(encoding='utf-8')))

    def puntajes(self, mediciones, sexo):
        """Lista de (edad, Z corregido) de las mediciones válidas: edad de 0 a 18 meses, talla de 40 a 125 cm
        y |Z| <= 6 antes de la corrección. A cada puntaje se le resta el sesgo del carné."""
        sexo = str(sexo).upper()
        if sexo not in self._lms:
            raise ValueError("sexo debe ser 'M' o 'F'")
        tabla = self._lms[sexo]
        salida = []
        for edad, talla in mediciones:
            edad, talla = float(edad), float(talla)
            if not (0 <= edad <= EDAD_MAX_PREDICTORES and RANGO_TALLA_CM[0] <= talla <= RANGO_TALLA_CM[1]):
                continue
            L, M, S = tabla[mes_lms(edad)]
            z = puntaje_z(talla, L, M, S)
            if abs(z) > Z_MAX:
                continue
            salida.append((edad, z - self.sesgo))
        return salida

    def calcular_rasgos(self, mediciones, sexo):
        """Variables de trayectoria, o None si hay menos de tres mediciones válidas."""
        p = self.puntajes(mediciones, sexo)
        if len(p) < MIN_MEDICIONES:
            return None
        edades, z = zip(*p)
        return rasgos_trayectoria(edades, z)

    def probabilidad_desde_rasgos(self, rasgos):
        """Probabilidad a partir de un diccionario (o fila) con las variables de `RASGOS`."""
        x = np.array([np.nan if rasgos[r] is None else float(rasgos[r]) for r in self.rasgos], dtype=float)
        x = np.where(np.isnan(x), self._mediana, x)
        eta = self._intercepto + float(np.dot(self._coef, (x - self._media) / self._escala))
        return 1.0 / (1.0 + math.exp(-eta))

    def probabilidad(self, mediciones, sexo):
        """Probabilidad estimada de DCI entre los 24 y 42 meses.

        mediciones: lista de pares (edad en meses, longitud en cm); se usan las tomadas hasta los 18 meses.
        sexo: 'M' (masculino) o 'F' (femenino).
        Devuelve None si hay menos de tres mediciones válidas.
        """
        f = self.calcular_rasgos(mediciones, sexo)
        return None if f is None else self.probabilidad_desde_rasgos(f)

    def evaluar(self, mediciones, sexo, umbral=None):
        """Probabilidad, clasificación según el umbral y variables usadas."""
        umbral = self.umbral if umbral is None else umbral
        f = self.calcular_rasgos(mediciones, sexo)
        if f is None:
            return {'probabilidad': None, 'riesgo_alto': None, 'umbral': umbral,
                    'mensaje': f'Se necesitan al menos {MIN_MEDICIONES} mediciones válidas hasta los '
                               f'{EDAD_MAX_PREDICTORES} meses.'}
        p = self.probabilidad_desde_rasgos(f)
        return {'probabilidad': p, 'riesgo_alto': None if umbral is None else bool(p >= umbral),
                'umbral': umbral, 'rasgos': {k: f[k] for k in self.rasgos}}


def _leer_mediciones(textos):
    salida = []
    for t in textos:
        edad, talla = t.replace(',', '.').split(':') if t.count(':') == 1 else (None, None)
        if edad is None:
            raise argparse.ArgumentTypeError(f'formato inválido: {t!r}; use edad:talla, por ejemplo 6:67.0')
        salida.append((float(edad), float(talla)))
    return salida


def main(argv=None):
    ap = argparse.ArgumentParser(description='Probabilidad de DCI entre los 24 y 42 meses a partir del carné de control.')
    ap.add_argument('--sexo', required=True, choices=['M', 'F', 'm', 'f'], help='M: masculino; F: femenino')
    ap.add_argument('--modelo', default=None, help='ruta a parametros_modelo.json')
    ap.add_argument('--umbral', type=float, default=None, help='umbral de clasificación (por defecto, el del modelo)')
    ap.add_argument('mediciones', nargs='+', help='mediciones edad_meses:longitud_cm, por ejemplo 2:57.5 6:67 12:75.5')
    a = ap.parse_args(argv)
    modelo = ModeloRiesgoDCI.cargar(a.modelo)
    r = modelo.evaluar(_leer_mediciones(a.mediciones), a.sexo, a.umbral)
    if r['probabilidad'] is None:
        print(r['mensaje'])
        return 1
    print(f"Probabilidad estimada de DCI a los 24-42 meses: {r['probabilidad']:.1%}")
    if r['riesgo_alto'] is not None:
        print(f"Clasificación (umbral {r['umbral']:.2f}): {'riesgo alto' if r['riesgo_alto'] else 'riesgo bajo'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
