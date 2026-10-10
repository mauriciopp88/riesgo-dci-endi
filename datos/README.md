# Datos

Esta carpeta no se versiona salvo este archivo y la tabla LMS. Coloque aquí:

| Archivo | Origen |
|---|---|
| `BDD_ENDI_R2_f1_personas.dta` | ENDI Ronda 2 (2023-2024), catálogo ANDA del INEC |
| `BDD_ENDI_R2_f2_salud_ninez.dta` | ENDI Ronda 2 (2023-2024), catálogo ANDA del INEC |
| `lms_talla_edad_oms.csv` | patrón de crecimiento de la OMS (ver abajo) |

Los microdatos se descargan de <https://anda.inec.gob.ec/anda5/index.php/catalog/1106> y se usan conforme a las
condiciones del INEC; no se redistribuyen en este repositorio. Si prefiere guardarlos en otra carpeta, indíquela
en la variable de entorno `ENDI_DATOS`.

## Tabla LMS (`lms_talla_edad_oms.csv`)

Parámetros LMS de longitud/talla para la edad del patrón de crecimiento infantil de la OMS (WHO Multicentre Growth
Reference Study Group, 2006), por sexo y mes cumplido:

| Columna | Descripción |
|---|---|
| `sexo` | `M` (masculino) o `F` (femenino) |
| `mes` | edad en meses, de 0 a 60 |
| `L`, `M`, `S` | parámetros del método LMS; el puntaje *Z* es ((talla / M) ** L − 1) / (L × S) |

La OMS publica dos tablas por sexo, longitud de 0 a 24 meses y talla de 24 a 60 meses, así que el mes 24
aparece dos veces. El notebook conserva la última fila de cada sexo y mes, por lo que la fila de talla debe ir
después de la de longitud. Tras esa depuración quedan 122 filas (2 sexos × 61 meses); el notebook lo verifica.

Para construirla desde las tablas oficiales
(<https://www.who.int/tools/child-growth-standards/standards/length-height-for-age>), descargue las cuatro tablas
mensuales de puntajes z (niños y niñas; 0 a 2 y 2 a 5 años) y ejecute:

```bash
python scripts/construir_lms_oms.py --ninos-0-2 <archivo> --ninos-2-5 <archivo> \
                                    --ninas-0-2 <archivo> --ninas-2-5 <archivo>
```

Los parámetros del patrón OMS son públicos, por lo que esta tabla sí puede versionarse junto con el código.
