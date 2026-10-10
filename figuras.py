# -*- coding: utf-8 -*-
"""Figuras del estudio (300 dpi, coma decimal, tipografía serif).

- figura_flujo: diagrama de selección de las muestras, a partir del diccionario de conteos del notebook.
- figura_bland_altman: carné frente a medición estandarizada, en dos paneles (resultados/datos/datos_bland_altman.csv).
- figura_roc_calibracion: curvas ROC y calibración por deciles (resultados/datos/predicciones_cv.csv).

También puede ejecutarse desde la terminal para regenerar las figuras a partir de los archivos exportados:
    python figuras.py resultados
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
                     'font.size': 9, 'axes.linewidth': 0.8, 'mathtext.fontset': 'dejavuserif'})

coma1 = FuncFormatter(lambda v, _: f'{v:.1f}'.replace('.', ',').replace('-', '−'))


def num(x, d=3):
    return f'{x:.{d}f}'.replace('.', ',').replace('-', '−')


def miles(n):
    return f'{int(n):,}'.replace(',', '.')


def pct(x, d=1):
    return f'{100 * x:.{d}f}%'.replace('.', ',')


def figura_flujo(F, ruta):
    """F: diccionario con los conteos del flujo de selección (sección 6 del notebook)."""
    fig, ax = plt.subplots(figsize=(6.3, 6.9), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(9, 116)
    ax.axis('off')

    def caja(xc, yc, w, h, texto, negrita=False, fs=8.2):
        ax.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h, boxstyle='round,pad=0.6,rounding_size=1.2',
                                    linewidth=1.0, edgecolor='black', facecolor='white', zorder=2))
        ax.text(xc, yc, texto, ha='center', va='center', fontsize=fs, zorder=3, linespacing=1.3,
                fontweight='bold' if negrita else 'normal')

    def flecha(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', lw=0.9, color='black', mutation_scale=10), zorder=1)

    def exclusion(x, y, texto, lado):
        ax.text(x + (-1.6 if lado == 'izq' else 1.6), y, texto, ha='right' if lado == 'izq' else 'left',
                va='center', fontsize=7.2, style='italic', linespacing=1.25)

    # tronco común
    caja(50, 110, 64, 8, 'ENDI Ronda 2 (2023-2024)\nNiños de 0 a 59 meses con estado nutricional válido: '
                         f'n = {miles(F["n_ninos"])}', True)
    flecha(50, 105.4, 50, 99.6)
    caja(38, 95, 52, 8, f'Niños con al menos una talla transcrita del carné\n'
                        f'n = {miles(F["n_carne"])} ({miles(F["med_carne"])} mediciones)')
    caja(84, 87.5, 29, 13.5, f'Mediciones excluidas\nEdad o talla fuera de rango: {miles(F["exc_rango"])}\n'
                             f'Sin sexo o parámetros LMS: {miles(F["exc_lms"])}\n|Z| > 6: {miles(F["exc_z6"])}', fs=7.4)
    flecha(38, 87.5, 68.8, 87.5)
    flecha(38, 90.4, 38, 84.6)
    caja(38, 80, 52, 8, f'Mediciones válidas del carné: {miles(F["med_validas"])}\n'
                        f'Niños con al menos una medición válida: n = {miles(F["n_validos"])}')
    # bifurcación
    ax.plot([38, 38], [75.4, 72], color='black', lw=0.9)
    ax.plot([24, 76], [72, 72], color='black', lw=0.9)
    ax.text(24, 73.4, 'Calidad del registro', ha='center', fontsize=8, style='italic')
    ax.text(76, 73.4, 'Modelo predictivo', ha='center', fontsize=8, style='italic')

    # rama A: calidad del registro
    flecha(24, 72, 24, 58.1)
    caja(24, 52, 42, 11, f'Con medición estandarizada válida\ny al menos una medición válida\n'
                         f'del carné: n = {miles(F["n_a_ambas"])}')
    flecha(24, 46.2, 24, 33.9)
    exclusion(24, 40.2, f'Excluidos: {miles(F["n_a_excl"])}\n(última medición\ndel carné fuera de\n'
                        f'la ventana de −3 a\n+1 meses)', 'izq')
    caja(24, 29, 42, 9, f'Muestra de calibración\nn = {miles(F["n_calibracion"])}', True)

    # rama B: modelo predictivo
    flecha(76, 72, 76, 58.1)
    exclusion(76, 65.0, f'Excluidos: {miles(F["n_b_excl_ventana"])}\n(fuera de 24–42 meses,\nsin resultado o sin\n'
                        f'mediciones hasta\nlos 18 meses)', 'der')
    caja(76, 52, 42, 11, f'De 24 a 42 meses, con resultado\nnutricional y alguna medición del\n'
                         f'carné hasta los 18 meses: n = {miles(F["n_b_ventana"])}')
    flecha(76, 46.2, 76, 33.9)
    exclusion(76, 40.2, f'Excluidos: {miles(F["n_b_excl_menos3"])}\n(menos de tres\nmediciones hasta\n'
                        f'los 18 meses)', 'der')
    caja(76, 29, 42, 9, f'Muestra de modelado\nn = {miles(F["n_modelado"])}; DCI: {pct(F["prev_modelado"])}', True)

    n18 = F['n_solapamiento_18m']
    quienes = 'Ninguno integra' if n18 == 0 else f'{miles(n18)} de ellos integra{"n" if n18 > 1 else ""}'
    ax.text(50, 16, f'Ambas muestras comparten {miles(F["n_solapamiento"])} niños '
                    f'({pct(F["n_solapamiento"] / F["n_modelado"])} de la muestra de modelado). {quienes} la\n'
                    f'submuestra con medición del carné hasta los 18 meses (n = {miles(F["n_sesgo_18m"])}), '
                    f'con la que se estimó\nel sesgo que corrige las variables del modelo.',
            ha='center', va='center', fontsize=7.4, style='italic', linespacing=1.3)
    fig.savefig(ruta, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    plt.close(fig)


def figura_bland_altman(csv, ruta):
    """(a) diferencia frente al promedio de ambas mediciones (Bland-Altman);
    (b) diferencia frente a la medición de referencia, más informativa cuando la referencia es mucho más
    precisa que el método evaluado (Krouwer, 2008)."""
    b = pd.read_csv(csv)
    m = (b.z_carne + b.z_endi) / 2
    d = b.z_carne - b.z_endi
    s, de = d.mean(), d.std()
    reg = np.polyfit(b.z_endi, d, 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 3.1), dpi=300, sharey=True)
    for ax, x in ((a1, m), (a2, b.z_endi)):
        ax.scatter(x, d, s=2.5, alpha=0.2, color='black', linewidths=0, rasterized=True)
        ax.axhline(0, color='gray', linewidth=0.6, linestyle=':')
        ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
        ax.xaxis.set_major_formatter(coma1)
        ax.yaxis.set_major_formatter(coma1)
        ax.spines[['top', 'right']].set_visible(False)
    a1.set_ylim(d.min() - 0.3, d.max() + 0.3)

    for yv, est, lab in [(s, '-', f'Sesgo medio: {num(s, 2)}'),
                         (s + 1.96 * de, '--', f'+1,96 DE: {num(s + 1.96 * de, 2)}'),
                         (s - 1.96 * de, '--', f'−1,96 DE: {num(s - 1.96 * de, 2)}')]:
        a1.axhline(yv, linestyle=est, color='black', linewidth=0.9)
        a1.text(m.max(), yv + 0.08, lab, ha='right', va='bottom', fontsize=6.6,
                bbox=dict(facecolor='white', edgecolor='none', pad=0.4, alpha=0.85))
    a1.set_xlabel('Promedio del puntaje Z (carné y encuesta)')
    a1.set_ylabel('Diferencia (carné − encuesta), DE')
    a1.set_title('(a)', loc='left', fontsize=9)

    xs = np.linspace(b.z_endi.min(), b.z_endi.max(), 100)
    a2.axhline(s, linestyle='-', color='black', linewidth=0.9)
    a2.plot(xs, np.polyval(reg, xs), linestyle='--', color='black', linewidth=1.1)
    a2.text(b.z_endi.max(), np.polyval(reg, b.z_endi.max()) - 0.25, f'Pendiente: {num(reg[0], 3)}',
            ha='right', va='top', fontsize=6.6,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.4, alpha=0.85))
    a2.set_xlabel('Puntaje Z de la medición estandarizada')
    a2.set_title('(b)', loc='left', fontsize=9)

    fig.tight_layout(w_pad=1.2)
    fig.savefig(ruta, dpi=300, facecolor='white')
    plt.close(fig)


def figura_roc_calibracion(csv, ruta):
    e = pd.read_csv(csv)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.6, 2.85), dpi=300)
    series = [('p_A', 'A. Sociodemográfico', (0, (1, 1.5))), ('p_B', 'B. Una medición previa', (0, (5, 2, 1, 2))),
              ('p_C', 'C. Trayectoria completa', '-'), ('p_D', 'D. Trayectoria + sociodem.', (0, (4, 2)))]
    for col, nom, est in series:
        fpr, tpr, _ = roc_curve(e.y, e[col])
        a1.plot(fpr, tpr, linestyle=est, color='black', linewidth=1.0,
                label=f'{nom} (AUC = {num(roc_auc_score(e.y, e[col]))})')
    a1.plot([0, 1], [0, 1], color='gray', linewidth=0.6)
    a1.set_xlabel('1 − especificidad')
    a1.set_ylabel('Sensibilidad')
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.01)
    a1.legend(frameon=False, fontsize=6.0, loc='lower right', handlelength=2.4, borderaxespad=0.2)
    a1.set_title('(a)', loc='left', fontsize=9)

    t = e.groupby(pd.qcut(e.p_C, 10, labels=False)).agg(pred=('p_C', 'mean'), obs=('y', 'mean'))
    a2.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=0.8, label='Calibración perfecta')
    a2.plot(t.pred, t.obs, marker='o', color='black', linewidth=1.0, markersize=3.5, label='Modelo final (deciles)')
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1)
    a2.set_xlabel('Probabilidad media predicha')
    a2.set_ylabel('Frecuencia observada de DCI')
    a2.legend(frameon=False, fontsize=6.2, loc='upper left')
    a2.set_title('(b)', loc='left', fontsize=9)
    for a in (a1, a2):
        a.xaxis.set_major_formatter(coma1)
        a.yaxis.set_major_formatter(coma1)
        a.spines[['top', 'right']].set_visible(False)
        a.set_aspect('equal')
    fig.tight_layout(w_pad=1.8)
    fig.savefig(ruta, dpi=300, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    res = Path(sys.argv[1] if len(sys.argv) > 1 else 'resultados')
    (res / 'figuras').mkdir(parents=True, exist_ok=True)
    cifras = json.loads((res / 'cifras.json').read_text(encoding='utf-8'))
    flujo = {k.split('.', 1)[1]: v for k, v in cifras.items() if k.startswith('flujo.')}
    figura_flujo(flujo, res / 'figuras' / 'figura1_flujo.png')
    figura_bland_altman(res / 'datos' / 'datos_bland_altman.csv', res / 'figuras' / 'figura2_bland_altman.png')
    figura_roc_calibracion(res / 'datos' / 'predicciones_cv.csv', res / 'figuras' / 'figura3_roc_calibracion.png')
    print('figuras guardadas en', (res / 'figuras').resolve())
