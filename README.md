# Riesgo de desnutrición crónica infantil a partir del carné de control (ENDI R2, Ecuador)

Código para estimar el riesgo de desnutrición crónica infantil (DCI) entre los 24 y 42 meses a partir de las
mediciones de longitud del carné de control del niño sano tomadas hasta los 18 meses, y para caracterizar el
sesgo y el error de esas mediciones frente a la medición estandarizada de la Encuesta Nacional sobre
Desnutrición Infantil (ENDI), Ronda 2 (2023-2024), del INEC.

## Contenido

```
├── notebooks/riesgo_dci_endi.ipynb    análisis completo: muestra, calidad del carné, modelos, validación y exportación
├── riesgo_dci.py                      módulo de inferencia (solo requiere NumPy)
├── figuras.py                         figuras a partir de los resultados exportados
├── scripts/construir_lms_oms.py       construye la tabla LMS a partir de las tablas oficiales de la OMS
├── pruebas/                           prueba de humo con datos sintéticos
├── datos/                             microdatos y tabla LMS (los microdatos no se incluyen; ver datos/README.md)
├── modelo/parametros_modelo.json      modelo final (lo genera el notebook)
└── resultados/                        tablas, figuras y cifras (las genera el notebook)
```

## Instalación

Con Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install jupyterlab             # o cualquier entorno que ejecute notebooks
```

## Datos

1. **Microdatos de la ENDI R2.** Descargue del
   [catálogo ANDA del INEC](https://anda.inec.gob.ec/anda5/index.php/catalog/1106), conforme a sus condiciones
   de uso, los archivos `BDD_ENDI_R2_f1_personas.dta` y `BDD_ENDI_R2_f2_salud_ninez.dta`, y cópielos en `datos/`.
   También puede dejarlos en otra carpeta e indicarla en la variable de entorno `ENDI_DATOS`.
2. **Tabla LMS de la OMS.** `datos/lms_talla_edad_oms.csv`, con los parámetros de longitud/talla para la edad del
   patrón de crecimiento de la OMS por sexo y mes (0 a 60). En `datos/README.md` se describe su formato y cómo
   construirla con `scripts/construir_lms_oms.py`.

## Ejecución

```bash
jupyter lab notebooks/riesgo_dci_endi.ipynb
```

Ejecute todas las celdas. El notebook toma unos minutos; la parte más lenta es el bootstrap de conglomerados.
Genera:

| Salida | Contenido |
|---|---|
| `resultados/tablas/*.csv` | características de la muestra, calidad del carné, prevalencias, desempeño, calibración, reglas de clasificación, coeficientes del modelo y verificación de cifras |
| `resultados/figuras/*.png` | diagrama de flujo, gráfico de Bland-Altman, curvas ROC y calibración (300 dpi) |
| `resultados/cifras.json` | todas las cifras calculadas |
| `resultados/datos/*.csv` | predicciones de validación cruzada y pares carné-encuesta, sin identificadores |
| `modelo/parametros_modelo.json` | parámetros del modelo final para el módulo de inferencia |

La última sección compara las cifras obtenidas con los valores de referencia, redondeados como se informaron.
Con los microdatos de la ENDI R2 y las versiones de `requirements.txt`, todas deben coincidir.

## Uso del modelo

El modelo recibe el sexo del niño y al menos tres mediciones del carné tomadas hasta los 18 meses, cada una como
(edad en meses, longitud en cm). A cada puntaje *Z* le resta el sesgo del carné, un valor único de 0,450 DE
estimado en las mediciones hasta los 18 meses; luego construye las variables de trayectoria y devuelve la
probabilidad estimada de DCI entre los 24 y 42 meses.

```python
from riesgo_dci import ModeloRiesgoDCI

modelo = ModeloRiesgoDCI.cargar()          # modelo/parametros_modelo.json
modelo.probabilidad([(2, 57.5), (6, 67.0), (12, 75.5), (17, 81.0)], sexo='M')   # 0,034
modelo.probabilidad([(2, 54.0), (6, 62.0), (12, 68.0), (17, 71.5)], sexo='M')   # 0,951
modelo.evaluar([(2, 54.0), (6, 62.0), (12, 68.0), (17, 71.5)], sexo='M')        # probabilidad, clasificación y variables
```

Desde la terminal:

```bash
python riesgo_dci.py --sexo M 2:54 6:62 12:68 17:71.5
```

Se descartan las mediciones fuera de 0 a 18 meses, de 40 a 125 cm o con |*Z*| > 6, igual que en el
entrenamiento. Con el umbral de 0,20 que usa el módulo, en validación cruzada agrupada el modelo detectó el 81,5%
de los casos con una especificidad de 76,4%.

**Alcance.** El modelo se entrenó con niños ecuatorianos de la ENDI R2 que tenían al menos tres mediciones en el
carné, una muestra con más DCI que la población general. La probabilidad sirve para priorizar el seguimiento; no
es un diagnóstico ni reemplaza la medición estandarizada. La corrección del sesgo es una constante, y el sesgo
real es mayor en los niños más pequeños y de menor talla.

## Reproducibilidad

- Las versiones de los paquetes están fijadas en `requirements.txt`. La asignación de conglomerados a pliegues de
  `GroupKFold` cambió entre scikit-learn 1.8 y 1.9; con otra versión, las métricas de validación cruzada pueden
  diferir en el tercer decimal.
- Semillas: 2026 para el bootstrap y 42 para Random Forest y Gradient Boosting.
- Prueba de humo, que ejecuta el notebook completo sobre datos sintéticos en una carpeta temporal:

  ```bash
  pip install nbclient ipykernel
  python pruebas/prueba_sintetica.py
  ```

## Privacidad

El repositorio no contiene microdatos. Los archivos exportados no tienen identificadores: el conglomerado se
recodifica como un índice. Aun así, `resultados/datos/`, con datos individuales derivados, queda fuera del
control de versiones por defecto (`.gitignore`).

## Cómo citar

[Se completará cuando el artículo asociado esté publicado.]
