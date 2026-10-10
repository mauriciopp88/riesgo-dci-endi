# -*- coding: utf-8 -*-
"""Prueba de humo: ejecuta el notebook completo sobre datos sintéticos y prueba el módulo de inferencia.

Copia el código a una carpeta temporal, de modo que no toca resultados/ ni modelo/ del repositorio.
Uso (desde la raíz del repositorio):
    python pruebas/prueba_sintetica.py
Requiere, además de requirements.txt, los paquetes nbclient e ipykernel.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'pruebas'))
from datos_sinteticos import generar  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for archivo in ['riesgo_dci.py', 'figuras.py']:
            shutil.copy(REPO / archivo, tmp / archivo)
        (tmp / 'notebooks').mkdir()
        generar(tmp / 'datos')
        print('datos sintéticos generados')

        nb = nbformat.read(REPO / 'notebooks' / 'riesgo_dci_endi.ipynb', as_version=4)
        t0 = time.time()
        NotebookClient(nb, timeout=1800, kernel_name='python3',
                       resources={'metadata': {'path': str(tmp / 'notebooks')}}).execute()
        print(f'notebook ejecutado sin errores en {time.time() - t0:.0f} s')

        esperados = ['resultados/cifras.json', 'modelo/parametros_modelo.json',
                     'resultados/datos/predicciones_cv.csv', 'resultados/datos/datos_bland_altman.csv',
                     'resultados/figuras/figura1_flujo.png', 'resultados/figuras/figura2_bland_altman.png',
                     'resultados/figuras/figura3_roc_calibracion.png',
                     'resultados/tablas/tabla_descriptiva.csv', 'resultados/tablas/tabla_calidad_carne.csv',
                     'resultados/tablas/tabla_desempeno.csv', 'resultados/tablas/tabla_reglas_clasificacion.csv',
                     'resultados/tablas/verificacion_cifras.csv']
        faltan = [a for a in esperados if not (tmp / a).exists()]
        assert not faltan, f'faltan salidas: {faltan}'
        cifras = json.loads((tmp / 'resultados' / 'cifras.json').read_text(encoding='utf-8'))
        print(f"salidas completas | {len(cifras)} cifras | AUC del modelo final (sintético): {cifras['desempeno.auc_C']:.3f}")

        # módulo de inferencia desde la terminal
        r = subprocess.run([sys.executable, 'riesgo_dci.py', '--sexo', 'M', '2:57.5', '6:67', '12:75.5', '17:81'],
                           cwd=tmp, capture_output=True, text=True, check=True)
        print('módulo desde la terminal:', r.stdout.strip().splitlines()[0])
        r = subprocess.run([sys.executable, 'riesgo_dci.py', '--sexo', 'F', '2:57.5', '6:67'],
                           cwd=tmp, capture_output=True, text=True)
        assert r.returncode == 1 and 'al menos 3' in r.stdout, 'con menos de 3 mediciones debe avisar'
    print('OK: la prueba de humo terminó sin errores')


if __name__ == '__main__':
    main()
