import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analisis import (ETIQUETAS, VARIABLES_MODELO, acp, cargar, evaluar_k,
                      kmeans, preparar, transformar)

st.set_page_config(page_title="Nasdaq-100 · ACP y K-Means", page_icon="📊", layout="wide")

COLORES = px.colors.qualitative.Bold
PLANTILLA = "plotly_white"


@st.cache_data
def datos():
    return cargar("data/nasdaq100.csv")


df_raw = datos()
FECHA = df_raw["fecha_corte"].iloc[0]

# ----------------------------------------------------------------- barra lateral
st.sidebar.title("Nasdaq-100")
st.sidebar.caption("Perfiles de riesgo y fundamentales")
pagina = st.sidebar.radio("Sección", [
    "1. Contexto", "2. Exploración de datos", "3. Preparación",
    "4. ACP", "5. K-Means", "6. Conclusiones"])

st.sidebar.divider()
st.sidebar.subheader("Parámetros del modelo")
variables = st.sidebar.multiselect(
    "Variables del modelo (mínimo 4)", VARIABLES_MODELO, default=VARIABLES_MODELO,
    format_func=lambda v: ETIQUETAS[v])
if len(variables) < 4:
    st.sidebar.error("Selecciona al menos 4 variables.")
    st.stop()

p_wins = st.sidebar.slider("Winsorización (% recortado en cada cola)", 0.0, 10.0, 5.0, 0.5) / 100

d, X, Z = preparar(df_raw, variables, p_wins)
pca, comps, cargas, var = acp(Z)

kaiser = int((var["Valor propio"] > 1).sum())
n_comp = st.sidebar.slider("Componentes a conservar", 2, len(variables), max(2, kaiser),
                           help=f"Criterio de Kaiser (valor propio > 1): {kaiser} componentes")
espacio = st.sidebar.radio("K-Means se aplica sobre", ["Componentes seleccionados", "Variables estandarizadas"])
X_km = comps.iloc[:, :n_comp] if espacio.startswith("Comp") else Z
k = st.sidebar.slider("Número de grupos (K)", 2, 8, 4)

km, etiquetas, sil_obs = kmeans(X_km, k)
d["Grupo"] = [f"Grupo {chr(65 + g)}" for g in etiquetas]
d["Silueta"] = sil_obs
orden_grupos = sorted(d["Grupo"].unique())

st.sidebar.divider()
st.sidebar.caption(f"Datos con corte al {FECHA}. Fuente: Nasdaq y Yahoo Finance.")


def describir_grupos(d, variables):
    """Describe cada grupo comparando su mediana con la del total (en desviaciones estándar)."""
    base = d[variables]
    z = (d.groupby("Grupo")[variables].median() - base.median()) / base.std()
    textos = {}
    for g, fila in z.iterrows():
        # el drawdown es negativo: un valor "alto" significa caídas más pequeñas
        altos = [ETIQUETAS[v].lower() for v in fila.index if fila[v] > 0.3 and v != "max_drawdown"]
        bajos = [ETIQUETAS[v].lower() for v in fila.index if fila[v] < -0.3 and v != "max_drawdown"]
        partes = []
        if altos:
            partes.append("más alto en " + ", ".join(altos))
        if bajos:
            partes.append("más bajo en " + ", ".join(bajos))
        if "max_drawdown" in fila.index and abs(fila["max_drawdown"]) > 0.3:
            partes.append("caídas máximas más " + ("pequeñas" if fila["max_drawdown"] > 0 else "profundas"))
        textos[g] = "; ".join(partes) if partes else "cercano al promedio en todas las variables"
    return textos, z


# =========================================================== 1. CONTEXTO
if pagina.startswith("1"):
    st.title("¿Todas las empresas del Nasdaq-100 son iguales?")
    st.markdown(f"""
El Nasdaq-100 reúne las 100 empresas no financieras más grandes que cotizan en Nasdaq. Se suele ver
como "el índice de las tecnológicas", y mucha gente lo opera (por ejemplo con el futuro **NQ** o el ETF **QQQ**)
como si fuera un bloque homogéneo. Pero dentro del índice conviven empresas muy distintas: desde Apple,
Costco o Pepsi hasta compañías de semiconductores o startups que todavía no son rentables.

**Pregunta del proyecto:** ¿qué perfiles de riesgo, desempeño y fundamentales existen dentro del Nasdaq-100,
y qué tan distintos son entre sí?

**¿Por qué sirve?** Si dentro del índice hay segmentos con comportamientos diferentes, un inversionista
puede diversificar mejor, entender qué está impulsando al índice y no asumir que "comprar tecnología" es
una sola apuesta.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Empresas analizadas", len(d))
    c2.metric("Variables cuantitativas", len(VARIABLES_MODELO))
    c3.metric("Corte de los datos", str(FECHA))

    st.subheader("Datos")
    st.markdown("""
- **Cada observación** es una empresa que hace parte del Nasdaq-100.
- **Composición del índice:** API pública de Nasdaq (101 títulos).
- **Precios diarios del último año y fundamentales:** Yahoo Finance, mediante la librería `yfinance`.
- **Variables de mercado** (calculadas por nosotros con los precios): rendimiento a 1 año, volatilidad anualizada,
  beta frente al índice Nasdaq-100 (^NDX) y máximo drawdown.
- **Variables fundamentales:** capitalización bursátil, P/E proyectado, margen neto y crecimiento de ingresos.
""")
    st.subheader("¿Por qué ACP y K-Means?")
    st.markdown("""
- Las variables de riesgo están muy relacionadas entre sí (volatilidad, beta y drawdown miden casi lo mismo).
  El **ACP** resume esa información redundante en pocas dimensiones y muestra cuáles son los ejes reales
  que diferencian a las empresas.
- **K-Means** busca grupos de empresas parecidas sin decirle de antemano cuáles son. Así se puede comparar
  si los grupos coinciden con los sectores tradicionales o si aparecen perfiles distintos.
- Las dos técnicas se complementan: primero se limpia el ruido y la redundancia con el ACP, y luego se agrupa
  en ese espacio más compacto.
""")
    st.subheader("Limitaciones de la fuente")
    st.markdown("""
- Las métricas de mercado son de **un solo año**: describen un momento del ciclo, no un rasgo permanente.
- Yahoo Finance no es una fuente auditada; algunos fundamentales pueden estar desactualizados.
- Dos empresas que acaban de entrar al índice (**SPCX** y **HONA**) se excluyeron porque no tienen un año de historia.
- La composición del índice cambia con el tiempo: este es un corte transversal a una fecha.
""")

# =========================================================== 2. EXPLORACIÓN
elif pagina.startswith("2"):
    st.title("Exploración de los datos")
    raw = df_raw.copy()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", raw.shape[0])
    c2.metric("Columnas", raw.shape[1])
    c3.metric("Tickers duplicados", int(raw["ticker"].duplicated().sum()))
    c4.metric("Celdas faltantes", int(raw.isna().sum().sum()))

    st.subheader("Base de datos original")
    st.dataframe(raw, width="stretch", height=300)

    st.subheader("Tipos de variables y faltantes")
    info = pd.DataFrame({"Tipo": raw.dtypes.astype(str), "Faltantes": raw.isna().sum()})
    st.dataframe(info, width="stretch")
    st.caption("Los faltantes están en ROE y deuda/patrimonio: corresponden a empresas con patrimonio negativo "
               "(Starbucks, O'Reilly, Booking, Marriott), donde esos indicadores no tienen sentido. "
               "Por eso esas dos variables no entran al modelo.")

    st.subheader("Estadísticas descriptivas")
    st.dataframe(raw.describe().T.style.format("{:,.3f}"), width="stretch")

    st.subheader("Composición por sector")
    sec = raw["sector"].value_counts().reset_index()
    fig = px.bar(sec, x="count", y="sector", orientation="h", template=PLANTILLA,
                 labels={"count": "Número de empresas", "sector": "Sector"},
                 title="Empresas del Nasdaq-100 por sector (Yahoo Finance)")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")
    st.caption("Tecnología pesa casi la mitad del índice, pero hay industriales, consumo, salud y servicios públicos.")

    st.subheader("Distribución de una variable")
    v = st.selectbox("Variable", ["market_cap", "pe_forward", "margen_neto", "crec_ingresos",
                                  "rendimiento_1a", "volatilidad_anual", "beta_ndx", "max_drawdown"])
    fig = px.histogram(raw, x=v, nbins=40, template=PLANTILLA, hover_data=["ticker"],
                       title=f"Distribución de {v} (datos sin transformar)")
    fig.update_layout(yaxis_title="Número de empresas")
    st.plotly_chart(fig, width="stretch")
    st.caption("Varias variables tienen colas muy largas (capitalización, P/E, rendimiento). Eso se trata en la sección de preparación.")

# =========================================================== 3. PREPARACIÓN
elif pagina.startswith("3"):
    st.title("Preparación de los datos")
    st.markdown("""
Decisiones tomadas antes de modelar, y por qué:

| Problema | Decisión | Razón |
|---|---|---|
| GOOG y GOOGL son la misma empresa | Se deja solo GOOGL | Si no, Alphabet pesaría el doble en el análisis |
| ROE y deuda/patrimonio con faltantes y valores absurdos (CoreWeave: 1.645%) | No entran al modelo | Con patrimonio negativo pierden sentido; imputarlos inventaría información |
| Capitalización entre 22 mil millones y 5 billones de USD | Logaritmo base 10 | Sin log, unas pocas megacaps dominarían todas las distancias |
| P/E negativo cuando hay pérdidas y enorme cuando la utilidad es casi cero (RKLB ≈ 860) | Se usa el **earnings yield** = 1 / P/E | Es continuo, se puede comparar entre empresas con y sin utilidades y mantiene la lectura (más alto = más barata) |
| Valores extremos (p. ej. un rendimiento de +800 %) | Winsorización en los percentiles elegidos en la barra lateral | ACP y K-Means usan varianzas y distancias, así que un solo extremo puede mover todo el resultado |
| Escalas distintas (porcentajes, betas, logaritmos) | Estandarización (media 0, desviación 1) | Para que ninguna variable pese más solo por sus unidades |
| Ticker, nombre, sector | No entran al modelo | Son identificadores o categorías; el sector se usa después para interpretar |
| ¿Normalizar cada fila a longitud 1 (como en el caso de tarjetas)? | **No** | Esa normalización compara el "estilo" y borra la magnitud. Aquí la magnitud importa: no es lo mismo una volatilidad del 30 % que una del 100 % |
""")
    st.info(f"Configuración actual: {len(variables)} variables y winsorización del {p_wins*100:.1f} % en cada cola.")

    st.subheader("Matriz de correlaciones (variables preparadas)")
    corr = X.corr().rename(index=ETIQUETAS, columns=ETIQUETAS)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    template=PLANTILLA, title="Correlación de Pearson entre variables del modelo", aspect="auto")
    fig.update_layout(height=550)
    st.plotly_chart(fig, width="stretch")
    st.markdown("""
**Cómo se lee:** volatilidad, beta, crecimiento de ingresos y drawdown forman un bloque muy correlacionado
(las empresas que más crecen también son las que más se mueven y más caen desde sus máximos). Tamaño, margen
y earnings yield se relacionan entre ellas, pero casi nada con el bloque de riesgo. Esa estructura es
justo lo que el ACP aprovecha para resumir.
""")

    st.subheader("Antes y después de preparar")
    v = st.selectbox("Variable", variables, format_func=lambda x: ETIQUETAS[x])
    c1, c2 = st.columns(2)
    antes = transformar(df_raw)[v]
    c1.plotly_chart(px.box(antes, template=PLANTILLA, title="Antes de winsorizar",
                           labels={"value": ETIQUETAS[v], "variable": ""}), width="stretch")
    c2.plotly_chart(px.box(Z[v], template=PLANTILLA, title="Winsorizada y estandarizada",
                           labels={"value": "Puntaje z", "variable": ""}), width="stretch")

# =========================================================== 4. ACP
elif pagina.startswith("4"):
    st.title("Análisis de Componentes Principales")

    st.subheader("Varianza explicada")
    st.dataframe(var.style.format({"Valor propio": "{:.3f}", "Varianza explicada": "{:.1%}", "Acumulada": "{:.1%}"}),
                 width="stretch", hide_index=True)

    fig = go.Figure()
    fig.add_bar(x=var["Componente"], y=var["Varianza explicada"], name="Por componente", marker_color=COLORES[0])
    fig.add_scatter(x=var["Componente"], y=var["Acumulada"], name="Acumulada", mode="lines+markers",
                    marker_color=COLORES[1])
    fig.add_vline(x=n_comp - 0.5, line_dash="dash", line_color="gray",
                  annotation_text=f"Se conservan {n_comp}")
    fig.update_layout(template=PLANTILLA, title="Gráfico de sedimentación (scree plot)",
                      yaxis_title="Proporción de varianza", yaxis_tickformat=".0%", xaxis_title="Componente")
    st.plotly_chart(fig, width="stretch")

    acum = var["Acumulada"].iloc[n_comp - 1]
    st.markdown(f"""
**¿Cuántos componentes conservar?** Con el criterio de Kaiser (valor propio mayor que 1) salen **{kaiser}**
componentes. En el gráfico se ve que después de ese punto cada componente aporta poco. Con **{n_comp}**
componentes se conserva el **{acum:.1%}** de la variación original. Si se exigiera el umbral habitual del 80 %
harían falta {int((var["Acumulada"] < 0.8).sum()) + 1} componentes. Se prefiere el criterio de Kaiser porque el cuarto
componente explica menos que una sola variable original (valor propio < 1), así que agrega poco.
""")

    st.subheader("Cargas de las variables")
    car = cargas.iloc[:, :n_comp].rename(index=ETIQUETAS)
    fig = px.imshow(car, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    template=PLANTILLA, title="Cargas: peso de cada variable en cada componente")
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")
    if len(variables) == len(VARIABLES_MODELO):
        st.markdown("""
**Interpretación (con las 8 variables):**
- **CP1, riesgo y crecimiento agresivo:** cargas altas en volatilidad, beta y crecimiento de ingresos, y negativa en
  drawdown (que es negativo, así que significa caídas más profundas). Un valor alto en CP1 es una acción que crece rápido,
  se mueve mucho más que el índice y cae fuerte.
- **CP2, calidad y madurez:** margen neto, tamaño y earnings yield. Separa empresas grandes, rentables y razonablemente
  valoradas de las pequeñas o con pérdidas.
- **CP3, valor contra momentum:** carga positiva en earnings yield y negativa en rendimiento y tamaño. Un valor alto es una
  empresa barata a la que le fue mal en el año; uno bajo, una ganadora cara.

El signo de un componente es arbitrario: lo que importa es qué variables van en la misma dirección.
""")
    else:
        st.caption("Interpreta cada componente según las variables con cargas más grandes en valor absoluto.")

    st.subheader("Empresas en el espacio de componentes")
    c1, c2 = st.columns(2)
    ex = c1.selectbox("Eje X", comps.columns[:n_comp], index=0)
    ey = c2.selectbox("Eje Y", comps.columns[:n_comp], index=1)
    vis = comps[[ex, ey]].copy()
    vis["ticker"], vis["sector"], vis["nombre"] = d["ticker"], d["sector"], d["nombre"]
    pct = var.set_index("Componente").loc[[ex, ey], "Varianza explicada"]
    fig = px.scatter(vis, x=ex, y=ey, color="sector", text="ticker", hover_name="nombre", template=PLANTILLA,
                     color_discrete_sequence=COLORES,
                     title=f"{ex} vs {ey}: este plano resume el {pct.sum():.1%} de la variación",
                     labels={ex: f"{ex} ({pct[ex]:.1%})", ey: f"{ey} ({pct[ey]:.1%})"})
    # vectores de las cargas (biplot)
    esc = np.abs(vis[[ex, ey]]).max().max() * 0.9
    for v in cargas.index:
        fig.add_annotation(x=cargas.loc[v, ex] * esc, y=cargas.loc[v, ey] * esc, ax=0, ay=0, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="gray",
                           text=ETIQUETAS[v], font=dict(size=10, color="dimgray"))
    fig.update_traces(textposition="top center", textfont_size=8, selector=dict(mode="markers+text"))
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")
    st.caption("Las flechas grises muestran hacia dónde apunta cada variable original. Colorear por sector deja ver que "
               "los sectores tradicionales no separan bien a las empresas: hay tecnológicas en todas las zonas del gráfico.")

# =========================================================== 5. K-MEANS
elif pagina.startswith("5"):
    st.title("Segmentación con K-Means")
    st.caption(f"K-Means aplicado sobre: {espacio.lower()}"
               + (f" ({n_comp} componentes)" if espacio.startswith("Comp") else ""))

    ev = evaluar_k(X_km)
    c1, c2 = st.columns(2)
    fig = px.line(ev, x="K", y="Inercia", markers=True, template=PLANTILLA, title="Método del codo")
    fig.add_vline(x=k, line_dash="dash", line_color="gray")
    fig.update_layout(yaxis_title="Inercia (suma de distancias al centroide²)")
    c1.plotly_chart(fig, width="stretch")
    fig = px.line(ev, x="K", y="Silueta", markers=True, template=PLANTILLA, title="Coeficiente de silueta promedio")
    fig.add_vline(x=k, line_dash="dash", line_color="gray")
    c2.plotly_chart(fig, width="stretch")

    mejor = int(ev.loc[ev["Silueta"].idxmax(), "K"])
    st.markdown(f"""
**¿Cómo se elige K?** La silueta más alta sale con **K = {mejor}**, pero con 2 grupos solo se separan las
acciones de alto riesgo del resto, y eso no aporta mucho. El codo no es muy marcado: la inercia baja fuerte
hasta K = 4 y después baja más despacio. Por eso se propone **K = 4**, que es un buen punto entre separación y
utilidad para interpretar. Su silueta (≈ 0,30) indica una estructura **moderada**: los grupos existen, pero
los bordes entre ellos no son nítidos. Con el control de la barra lateral se pueden probar otros valores.
""")
    st.metric("Silueta promedio con el K elegido", f"{sil_obs.mean():.3f}")

    st.subheader("Grupos en el plano CP1–CP2")
    vis = comps.iloc[:, :2].copy()
    vis["Grupo"], vis["ticker"], vis["nombre"], vis["sector"] = d["Grupo"], d["ticker"], d["nombre"], d["sector"]
    fig = px.scatter(vis, x="CP1", y="CP2", color="Grupo", text="ticker", hover_name="nombre",
                     hover_data=["sector"], category_orders={"Grupo": orden_grupos},
                     color_discrete_sequence=COLORES, template=PLANTILLA,
                     title=f"Grupos de K-Means proyectados en CP1 y CP2 ({var['Acumulada'].iloc[1]:.1%} de la variación)",
                     labels={"CP1": "CP1: riesgo y crecimiento agresivo", "CP2": "CP2: calidad y madurez"})
    fig.update_traces(textposition="top center", textfont_size=8)
    cent = vis.groupby("Grupo")[["CP1", "CP2"]].mean()
    fig.add_scatter(x=cent["CP1"], y=cent["CP2"], mode="markers+text", text=cent.index, name="Centroides",
                    marker=dict(symbol="x", size=16, color="black", line=dict(width=2)),
                    textposition="bottom center", textfont=dict(size=12, color="black"))
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")
    st.caption("Las X negras son el centro de cada grupo en este plano. Las letras de los grupos son solo nombres: "
               "no indican orden ni jerarquía.")

    st.subheader("Perfil de cada grupo con las variables originales (medianas)")
    st.caption("Se usan medianas y no promedios porque hay valores extremos (por ejemplo, un rendimiento de +800 %) que "
               "moverían mucho el promedio de un grupo pequeño.")
    orig = ["market_cap", "pe_forward", "margen_neto", "crec_ingresos", "rendimiento_1a",
            "volatilidad_anual", "beta_ndx", "max_drawdown"]
    perfil = d.groupby("Grupo")[orig].median()
    perfil.insert(0, "Empresas", d["Grupo"].value_counts())
    perfil["Silueta media"] = d.groupby("Grupo")["Silueta"].mean()
    perfil["market_cap"] = perfil["market_cap"] / 1e9
    perfil = perfil.rename(columns={"market_cap": "Cap. (miles de mill. USD)", "pe_forward": "P/E fwd",
                                    "margen_neto": "Margen neto", "crec_ingresos": "Crec. ingresos",
                                    "rendimiento_1a": "Rend. 1 año", "volatilidad_anual": "Volatilidad",
                                    "beta_ndx": "Beta", "max_drawdown": "Máx. drawdown"})
    tot = d[orig].median()
    st.dataframe(perfil.style.format({
        "Cap. (miles de mill. USD)": "{:,.0f}", "P/E fwd": "{:.1f}", "Margen neto": "{:.1%}",
        "Crec. ingresos": "{:.1%}", "Rend. 1 año": "{:.1%}", "Volatilidad": "{:.1%}", "Beta": "{:.2f}",
        "Máx. drawdown": "{:.1%}", "Silueta media": "{:.2f}"}), width="stretch")
    st.caption(f"Referencia, mediana del total: volatilidad {tot['volatilidad_anual']:.0%}, beta {tot['beta_ndx']:.2f}, "
               f"rendimiento {tot['rendimiento_1a']:.0%}, margen {tot['margen_neto']:.0%}.")

    textos, zg = describir_grupos(d, variables)
    fig = px.imshow(zg.rename(columns=ETIQUETAS).loc[orden_grupos], text_auto=".1f", color_continuous_scale="RdBu_r",
                    zmin=-2, zmax=2, aspect="auto", template=PLANTILLA,
                    title="Qué tan lejos está cada grupo de la mediana general (en desviaciones estándar)")
    st.caption("Rojo = por encima de la mediana general; azul = por debajo. En drawdown, rojo significa caídas más pequeñas. "
               "Se resaltan en el texto las diferencias mayores a 0,3 desviaciones.")
    st.plotly_chart(fig, width="stretch")
    for g in orden_grupos:
        ej = ", ".join(d[d["Grupo"] == g].sort_values("market_cap", ascending=False)["ticker"].head(8))
        st.markdown(f"**{g}** ({(d['Grupo'] == g).sum()} empresas): {textos[g]}. Ejemplos: {ej}.")

    st.subheader("Sectores dentro de cada grupo")
    ct = pd.crosstab(d["Grupo"], d["sector"])
    fig = px.bar(ct.reset_index().melt(id_vars="Grupo"), x="Grupo", y="value", color="sector",
                 template=PLANTILLA, color_discrete_sequence=px.colors.qualitative.Pastel,
                 labels={"value": "Número de empresas"}, title="Composición sectorial de cada grupo")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Gráfico de silueta por empresa")
    s = d.sort_values(["Grupo", "Silueta"])
    fig = px.bar(s, x="Silueta", y="ticker", color="Grupo", orientation="h", template=PLANTILLA,
                 color_discrete_sequence=COLORES, category_orders={"Grupo": orden_grupos},
                 title="Silueta de cada empresa (cerca de 0 o negativa = difícil de clasificar)")
    fig.update_layout(height=1400, yaxis={"categoryorder": "array", "categoryarray": s["ticker"].tolist()})
    fig.update_yaxes(tickfont_size=7)
    st.plotly_chart(fig, width="stretch")
    dudosas = d[d["Silueta"] < 0.1].sort_values("Silueta")
    st.markdown(f"**Empresas en la frontera** (silueta < 0,1): {', '.join(dudosas['ticker']) or 'ninguna'}. "
                "Están entre dos perfiles y su asignación no es tan confiable.")

    st.subheader("Buscar una empresa")
    t = st.selectbox("Ticker", sorted(d["ticker"]))
    fila = d[d["ticker"] == t].iloc[0]
    st.write(f"**{fila['nombre']}** ({fila['sector']}) quedó en el **{fila['Grupo']}**, con silueta {fila['Silueta']:.2f}.")

    st.subheader("Listado completo")
    salida = d[["ticker", "nombre", "sector", "Grupo", "Silueta"] + orig].sort_values(["Grupo", "ticker"])
    st.dataframe(salida, width="stretch", hide_index=True)
    st.download_button("Descargar la base con el grupo asignado (CSV)", salida.to_csv(index=False).encode("utf-8"),
                       file_name="nasdaq100_con_grupos.csv", mime="text/csv")

# =========================================================== 6. CONCLUSIONES
else:
    st.title("Discusión de resultados y conclusiones")
    if not (len(variables) == len(VARIABLES_MODELO) and k == 4 and n_comp == 3 and espacio.startswith("Comp")):
        st.warning("Este texto está escrito para la configuración base: las 8 variables, 3 componentes y K = 4 sobre los "
                   "componentes. Con otros parámetros los números de abajo cambian, pero el texto no.")
    acum3 = var["Acumulada"].iloc[2]
    st.markdown(f"""
### ¿Qué variables definen los componentes?
El primer componente (42 % de la variación) es básicamente **riesgo**: volatilidad, beta, drawdown y crecimiento de
ingresos van juntos. Ese es el eje que más diferencia a las empresas del Nasdaq-100. El segundo es **calidad y
tamaño** (margen, capitalización, earnings yield), y el tercero contrapone **valor contra momentum**.
Con tres componentes conservamos el **{acum3:.1%}** de la información de las 8 variables, así que la tabla se puede
leer en tres ejes sin perder mucho.

### ¿Qué grupos aparecen?
Con K = 4 salen cuatro perfiles con diferencias económicas claras:

1. **Núcleo estable (≈ 37 empresas):** Apple, Microsoft, Amazon, Meta, Walmart, Costco, Pepsi. Volatilidad baja (≈ 28 %),
   beta muy baja frente al índice y caídas pequeñas. El mercado les paga el P/E más alto: se paga una prima por estabilidad.
2. **Líderes del ciclo de IA y semiconductores (≈ 15):** Nvidia, Broadcom, Micron, AMD, ASML, Lam Research, Palantir.
   Grandes, muy rentables (margen ≈ 36 %), con crecimiento de ingresos alto y rendimiento de más del 50 % en el año,
   pero con una beta mayor que 2: son las que **mueven** al índice.
3. **Especulativas de alto riesgo (≈ 9):** CoreWeave, Nebius, Rocket Lab, Astera Labs, Intel, Arm. Crecen mucho,
   pero no son rentables (margen negativo) y tienen la volatilidad más alta (≈ 100 %) y los peores drawdowns.
4. **Rezagadas (≈ 37):** Netflix, Adobe, Intuit, PayPal, Qualcomm, T-Mobile y buena parte del software. Rendimiento
   cercano a 0 % en el año, P/E más bajo (más baratas) y caídas intermedias.

### ¿Los grupos están claramente separados?
No del todo. La silueta promedio ronda **0,30**, que indica una estructura moderada: las empresas se reparten más
como un **continuo** que como islas separadas. Hay empresas en la frontera (por ejemplo Marvell, con silueta negativa)
que podrían quedar en otro grupo con un cambio pequeño en los datos. La silueta más alta se da con K = 2, pero esa
partición solo dice "alto riesgo contra el resto", así que se prefirió una segmentación más informativa.

### ¿Qué aprendimos frente a la pregunta?
- **El Nasdaq-100 no es un bloque homogéneo.** Hay tecnológicas en los cuatro grupos, así que el sector no explica
  el perfil de riesgo: lo explica cómo se comporta y cómo gana dinero cada empresa.
- El índice del último año estuvo dominado por el bloque de semiconductores e IA. Las megacaps tradicionales como Apple
  o Microsoft se comportaron casi como defensivas frente al índice.
- Para alguien que opera NQ o QQQ: estar largo en el índice es, en buena parte, una apuesta concentrada en el
  grupo 2. Quien quiera diversificar dentro del mismo índice puede mirar el núcleo estable.

### Limitaciones
- **Un solo año de datos de mercado:** los grupos reflejan un momento del ciclo (el rally de IA). En otro año,
  las mismas empresas podrían caer en grupos distintos.
- La **beta se mide contra el mismo índice** del que hacen parte las empresas, y las más grandes pesan mucho en él.
- La winsorización y la elección de K son decisiones del analista; los resultados son robustos a cambios moderados
  en el recorte, pero no son la "verdad" sobre el mercado.
- Los fundamentales de Yahoo Finance no son auditados y mezclan periodos fiscales distintos.
- K-Means supone grupos más o menos redondos y obliga a cada empresa a pertenecer a alguno. Con una estructura tan
  continua, métodos como el **clustering jerárquico** o las **mezclas gaussianas** (que dan una probabilidad de
  pertenencia en vez de una etiqueta fija) podrían describir mejor las empresas de frontera.
""")
