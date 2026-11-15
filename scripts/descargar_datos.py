"""
Descarga los datos del proyecto y genera data/nasdaq100.csv

Fuentes:
- Composición del Nasdaq-100: API pública de Nasdaq (api.nasdaq.com)
- Precios diarios y fundamentales: Yahoo Finance mediante la librería yfinance

Uso:  python scripts/descargar_datos.py
"""
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# 1. Composición del índice
r = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100", headers=H, timeout=30)
filas = r.json()["data"]["data"]["rows"]
lista = pd.DataFrame(filas)[["symbol", "companyName"]]
tickers = lista["symbol"].str.replace(".", "-", regex=False).tolist()
print(f"{len(tickers)} tickers en el Nasdaq-100")

# 2. Precios de 1 año (más el índice de referencia ^NDX para calcular beta)
precios = yf.download(tickers + ["^NDX"], period="1y", auto_adjust=True, progress=False)["Close"]
ret = precios.pct_change().dropna(how="all")
ret_ndx = ret["^NDX"]

def metricas_mercado(t):
    p = precios[t].dropna()
    r_ = ret[t].dropna()
    if len(p) < 200:  # se exige casi un año completo de historia
        return None
    comun = r_.index.intersection(ret_ndx.dropna().index)
    beta = np.cov(r_[comun], ret_ndx[comun])[0, 1] / ret_ndx[comun].var()
    drawdown = (p / p.cummax() - 1).min()
    return {
        "rendimiento_1a": p.iloc[-1] / p.iloc[0] - 1,
        "volatilidad_anual": r_.std() * np.sqrt(252),
        "beta_ndx": beta,
        "max_drawdown": drawdown,
    }

# 3. Fundamentales
def fundamentales(t):
    for _ in range(3):
        try:
            i = yf.Ticker(t).info
            return {
                "nombre": i.get("shortName"),
                "sector": i.get("sector"),
                "market_cap": i.get("marketCap"),
                "pe_forward": i.get("forwardPE"),
                "margen_neto": i.get("profitMargins"),
                "crec_ingresos": i.get("revenueGrowth"),
                "roe": i.get("returnOnEquity"),
                "deuda_patrimonio": i.get("debtToEquity"),
            }
        except Exception:
            time.sleep(2)
    return {}

registros = []
for t in tickers:
    m = metricas_mercado(t)
    if m is None:
        print(f"  {t}: sin historia suficiente, se omite")
        continue
    registros.append({"ticker": t, **fundamentales(t), **m})

df = pd.DataFrame(registros)
df["fecha_corte"] = precios.index[-1].date()
df.to_csv("data/nasdaq100.csv", index=False)
print(df.shape)
