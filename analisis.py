"""Funciones de preparación, ACP y K-Means. Las usa app.py."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Nombres legibles de las variables del modelo
ETIQUETAS = {
    "log_market_cap": "Tamaño (log cap. bursátil)",
    "earnings_yield": "Earnings yield (1 / P/E fwd)",
    "margen_neto": "Margen neto",
    "crec_ingresos": "Crecimiento de ingresos",
    "rendimiento_1a": "Rendimiento 1 año",
    "volatilidad_anual": "Volatilidad anual",
    "beta_ndx": "Beta vs Nasdaq-100",
    "max_drawdown": "Máximo drawdown",
}
VARIABLES_MODELO = list(ETIQUETAS)


def cargar(ruta="data/nasdaq100.csv"):
    return pd.read_csv(ruta)


def transformar(df):
    """Crea las variables del modelo a partir de los datos crudos."""
    d = df.copy()
    d = d.drop_duplicates(subset="ticker")
    # GOOG y GOOGL son dos clases de acción de la misma empresa (Alphabet):
    # se deja solo una para no contar la misma compañía dos veces
    d = d[d["ticker"] != "GOOG"].reset_index(drop=True)
    # El tamaño de las empresas va de 22 mil millones a 5 billones: se usa logaritmo
    d["log_market_cap"] = np.log10(d["market_cap"])
    # El P/E es discontinuo cuando la utilidad es negativa o cercana a cero;
    # su inverso (earnings yield) es continuo y conserva el orden económico
    d["earnings_yield"] = 1 / d["pe_forward"]
    return d


def winsorizar(X, p=0.05):
    """Recorta cada variable entre los percentiles p y 1-p."""
    li, ls = X.quantile(p), X.quantile(1 - p)
    return X.clip(lower=li, upper=ls, axis=1)


def preparar(df, variables, p_wins=0.05):
    d = transformar(df)
    X = d[variables].copy()
    if p_wins > 0:
        X = winsorizar(X, p_wins)
    Z = StandardScaler().fit_transform(X)
    return d, X, pd.DataFrame(Z, columns=variables, index=d.index)


def acp(Z):
    pca = PCA().fit(Z)
    comps = pd.DataFrame(pca.transform(Z), index=Z.index,
                         columns=[f"CP{i+1}" for i in range(Z.shape[1])])
    cargas = pd.DataFrame(pca.components_.T, index=Z.columns, columns=comps.columns)
    var = pd.DataFrame({
        "Componente": comps.columns,
        "Valor propio": pca.explained_variance_,
        "Varianza explicada": pca.explained_variance_ratio_,
        "Acumulada": np.cumsum(pca.explained_variance_ratio_),
    })
    return pca, comps, cargas, var


def evaluar_k(X, k_min=2, k_max=10, semilla=42):
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=semilla).fit(X)
        filas.append({"K": k, "Inercia": km.inertia_,
                      "Silueta": silhouette_score(X, km.labels_)})
    return pd.DataFrame(filas)


def kmeans(X, k, semilla=42):
    km = KMeans(n_clusters=k, n_init=20, random_state=semilla).fit(X)
    return km, km.labels_, silhouette_samples(X, km.labels_)
