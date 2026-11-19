# ¿Todas las empresas del Nasdaq-100 son iguales?
### Perfiles de riesgo y fundamentales con ACP y K-Means

Taller individual de **Machine Learning aplicado a Economía y Finanzas** — Pontificia Universidad Javeriana Cali.
Autor: Juan David Vivas Idarraga.

**App desplegada:** https://nasdaq100-acp-kmeans.streamlit.app/  
**Video de sustentación:** https://youtu.be/-x-iEq-DTJg


---

## Contexto y pregunta

El Nasdaq-100 se suele ver como "el índice de las tecnológicas", y mucha gente lo opera (con el futuro NQ o el ETF QQQ)
como si fuera un solo bloque. Pero adentro conviven empresas muy distintas: Apple, Costco, Pepsi, fabricantes de chips y
startups que todavía no son rentables.

**Pregunta:** ¿qué perfiles de riesgo, desempeño y fundamentales existen dentro del Nasdaq-100, y qué tan distintos son?

**¿Por qué ACP y K-Means?** Las métricas de riesgo están muy correlacionadas entre sí (volatilidad, beta y drawdown
miden casi lo mismo). El ACP resume esa redundancia en pocos ejes que se pueden interpretar. Sobre esos ejes, K-Means
busca grupos de empresas parecidas sin usar el sector como guía, así que se puede comprobar si los sectores tradicionales
explican de verdad los perfiles.

## Datos

| | |
|---|---|
| Observación | Una empresa del Nasdaq-100 |
| Cobertura | 98 empresas (101 títulos del índice; se excluyen SPCX y HONA por tener menos de un año de historia y GOOG por ser la misma empresa que GOOGL) |
| Fecha de corte | 13 de noviembre de 2026 (precios del último año) |
| Fuentes | Composición: API pública de Nasdaq · Precios y fundamentales: Yahoo Finance (`yfinance`) |
| Archivo | `data/nasdaq100.csv` |

**Variables del modelo (8):**

| Variable | Cómo se obtiene |
|---|---|
| Tamaño | log10 de la capitalización bursátil |
| Earnings yield | 1 / P/E proyectado |
| Margen neto | Yahoo Finance |
| Crecimiento de ingresos | Yahoo Finance (interanual) |
| Rendimiento 1 año | Calculado con precios ajustados |
| Volatilidad anual | Desviación de los retornos diarios × √252 |
| Beta vs Nasdaq-100 | Cov(acción, ^NDX) / Var(^NDX) con retornos diarios |
| Máximo drawdown | Mayor caída desde un máximo en el año |

**Limitaciones de la fuente:** solo un año de datos de mercado (refleja un momento del ciclo), Yahoo Finance no es una fuente
auditada y los fundamentales mezclan periodos fiscales distintos. ROE y deuda/patrimonio se descargaron, pero no entran
al modelo porque tienen faltantes y valores sin sentido en empresas con patrimonio negativo.

## Metodología

1. **Preparación:** eliminar el duplicado GOOG; logaritmo de la capitalización; P/E → earnings yield (para manejar P/E
   negativos o gigantes); winsorización al 5 % en cada cola; estandarización.
2. **ACP:** varianza explicada, scree plot y cargas. Se conservan **3 componentes** (criterio de Kaiser), que explican
   el **76,2 %** de la variación.
   - CP1 (42 %): riesgo y crecimiento agresivo (volatilidad, beta, crecimiento, drawdown)
   - CP2 (18 %): calidad y madurez (margen, tamaño, earnings yield)
   - CP3 (16 %): valor contra momentum
3. **K-Means** sobre los 3 componentes, comparando K = 2…10 con el método del codo y la silueta. Se elige **K = 4**
   (silueta ≈ 0,30): K = 2 tiene más silueta, pero solo separa "alto riesgo contra el resto".

## Resultados principales

| Perfil | Empresas | Rasgos (medianas) | Ejemplos |
|---|---|---|---|
| Núcleo estable | 37 | Volatilidad 28 %, beta 0,19, drawdown −20 %, P/E 25 | AAPL, MSFT, AMZN, META, WMT, COST |
| Líderes IA / semiconductores | 15 | Margen 36 %, crecimiento 56 %, rendimiento +53 %, beta 2,2 | NVDA, AVGO, MU, AMD, ASML |
| Especulativas de alto riesgo | 9 | Margen negativo, volatilidad 100 %, drawdown −62 % | CRWV, NBIS, RKLB, ALAB, INTC |
| Rezagadas | 37 | Rendimiento −3 %, P/E 17, volatilidad 40 % | NFLX, ADBE, INTU, PYPL, QCOM |

- El Nasdaq-100 **no es homogéneo**: hay tecnológicas en los cuatro grupos, así que el sector no explica el perfil.
- El último año del índice estuvo impulsado por el bloque de IA y semiconductores; las megacaps clásicas se comportaron casi como defensivas.
- Los grupos se parecen más a un continuo que a islas separadas (silueta moderada), y hay empresas en la frontera, como MRVL.

## Estructura del repositorio

```
├── app.py                    # Aplicación Streamlit (6 secciones)
├── analisis.py               # Preparación, ACP y K-Means (funciones)
├── data/nasdaq100.csv        # Datos descargados
├── scripts/descargar_datos.py# Descarga y cálculo de métricas
├── requirements.txt
└── .streamlit/config.toml    # Tema visual
```

## Cómo ejecutarlo

```bash
git clone https://github.com/jdavidvivasidarraga/nasdaq100-acp-kmeans.git
cd nasdaq100-acp-kmeans
pip install -r requirements.txt
streamlit run app.py
```

Para volver a descargar los datos con la composición y los precios actuales (los resultados cambiarán):

```bash
python scripts/descargar_datos.py
```

En la barra lateral de la app se pueden cambiar las variables, el porcentaje de winsorización, el número de componentes,
el espacio donde se aplica K-Means y el número de grupos. Todo se recalcula en vivo.
