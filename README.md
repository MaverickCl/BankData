# Análisis de Marketing Bancario

Proyecto para analizar campañas de marketing bancario usando datos
históricos.\
Incluye limpieza de datos, análisis exploratorio y generación de
gráficos para entender qué factores influyen en la conversión.

------------------------------------------------------------------------

## Requisitos

-   Python 3.8+
-   pip

------------------------------------------------------------------------

## Instalación

Desde la carpeta del proyecto:

``` bash
python -m venv venv
venv\Scripts\activate   # Windows
# o
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

Asegúrate de tener el archivo:

``` text
data/raw/bank.csv
```

Separado por `;` (punto y coma).

------------------------------------------------------------------------

## Estructura

``` text
bank_marketing_ds/
├── data/
│   ├── raw/bank.csv
│   └── processed/bank_clean.csv
├── src/
│   ├── main.py
│   ├── load_data.py
│   ├── clean_data.py
│   ├── analysis.py
│   └── basic_target_analysis.py
├── reports/figures/
└── requirements.txt
```

------------------------------------------------------------------------

## Uso

### 1. Limpiar datos

``` bash
python src/main.py
```

-   Carga `bank.csv`
-   Limpia datos
-   Guarda `bank_clean.csv`

------------------------------------------------------------------------

### 2. Exploración básica

``` bash
python src/analysis.py
```

Muestra: - columnas - tipos - nulos - estadísticas

------------------------------------------------------------------------

### 3. Análisis completo de conversión

``` bash
python src/basic_target_analysis.py
```

-   Genera todos los gráficos
-   Guarda imágenes en `reports/figures/`
-   Analiza canal, mes, edad, ocupación, préstamos, etc.

------------------------------------------------------------------------

## Notas

-   El dataset debe llamarse `bank.csv`
-   Los gráficos se sobrescriben al volver a ejecutar
-   Todo el análisis corre desde la raíz del proyecto

------------------------------------------------------------------------

**Autor:** Diego Garrido Uribe\
**Año:** 2026

