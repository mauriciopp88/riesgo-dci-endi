# Riesgo de desnutrición crónica infantil a partir del carné de control (ENDI R2, Ecuador)

## Contenido
- notebooks/00_articulo_iccs_riesgo_dci_v2.ipynb: notebook reproducible (muestra, calibración del sesgo,
  variables de trayectoria, comparación de modelos, modelo final).
- models/modelo_riesgo_dci.joblib: modelo entrenado.

## Datos
Los microdatos de la ENDI R2 NO están en este repositorio. Se obtienen del catálogo ANDA-INEC:
https://anda.inec.gob.ec/anda5/index.php/catalog/1106

## Reproducir
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/00_articulo_iccs_riesgo_dci_v2.ipynb
