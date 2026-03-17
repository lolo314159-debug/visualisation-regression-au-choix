import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Analyse Quantitative Premium", layout="wide")

# --- BARRE LATÉRALE ---
st.sidebar.header("Source des données")
input_method = st.sidebar.radio("Choisir la méthode :", ("Fichier Excel", "Saisie Manuelle"))

selected_ticker = ""
display_name = ""

if input_method == "Fichier Excel":
    uploaded_file = st.sidebar.file_uploader("Charger Excel", type="xlsx")
    if uploaded_file:
        df_tickers = pd.read_excel(uploaded_file)
        cols = df_tickers.columns.tolist()
        t_col = st.sidebar.selectbox("Colonne Tickers", cols, index=cols.index('Ticker') if 'Ticker' in cols else 0)
        n_col = next((c for c in cols if "nom" in c.lower()), None)
        
        ticker_list = df_tickers[t_col].dropna().unique().tolist()
        selected_ticker = st.sidebar.selectbox("Action", ticker_list)
        if n_col:
            display_name = df_tickers[df_tickers[t_col] == selected_ticker][n_col].iloc[0]
else:
    selected_ticker = st.sidebar.text_input("Saisir un Ticker (ex: AAPL, AIR.PA)", "MSFT").upper()

regression_type = st.sidebar.radio("Modèle", ("Logarithmique", "Linéaire"))

# --- RÉCUPÉRATION ET CALCULS ---
if selected_ticker:
    with st.spinner(f'Analyse de {selected_ticker}...'):
        ticker_obj = yf.Ticker(selected_ticker)
        data = ticker_obj.history(start="2000-01-01")
        
        # Récupérer le nom via Yahoo Finance si pas trouvé dans l'Excel
        if not display_name:
            try: display_name = ticker_obj.info.get('longName', selected_ticker)
            except: display_name = selected_ticker

    if not data.empty and len(data) > 20:
        df = data[['Close']].copy()
        df = df.dropna().reset_index()
        
        # Maths
        df['Index'] = np.arange(len(df)).reshape(-1, 1)
        X = df['Index'].values.reshape(-1, 1)
        y_val = np.log(df['Close'].values) if regression_type == "Logarithmique" else df['Close'].values
        
        model = LinearRegression().fit(X, y_val)
        y_pred = model.predict(X)
        std_err = np.std(y_val - y_pred)
        
        def to_p(v): return np.exp(v) if regression_type == "Logarithmique" else v

        # --- GRAPHIQUE AVEC CANAUX PASTEL ---
        fig = go.Figure()

        # Zone 2-Sigma (Rouge Pastel très léger)
        fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred + 2*std_err), line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred - 2*std_err), fill='tonexty', 
                                 fillcolor='rgba(255, 0, 0, 0.05)', line=dict(width=0), name="Zone ±2σ (Extrême)"))

        # Zone 1-Sigma (Vert Pastel léger)
        fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred + std_err), line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred - std_err), fill='tonexty', 
                                 fillcolor='rgba(0, 255, 100, 0.1)', line=dict(width=0), name="Zone ±1σ (Canal)"))

        # Lignes de tendance et prix
        fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred), line=dict(color='rgba(255, 165, 0, 0.6)', width=1.5, dash='dash'), name="Tendance"))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='#FFFFFF', width=2.5), name="Prix de Clôture"))

        fig.update_layout(
            title=f"<b>{display_name}</b> <span style='color:gray;'>| {selected_ticker}</span>",
            title_font_size=26,
            template="plotly_dark",
            hovermode="x unified",
            yaxis_type="log" if regression_type == "Logarithmique" else "linear",
            paper_bgcolor='black', plot_bgcolor='black',
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', side="right", title="Prix"),
            xaxis=dict(showgrid=False)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- KPIs ---
        days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
        cagr = (df['Close'].iloc[-1] / df['Close'].iloc[0])**(365.25 / days) - 1
        vol = (np.log(df['Close'] / df['Close'].shift(1))).std() * np.sqrt(252)
        curr_y = np.log(df['Close'].iloc[-1]) if regression_type == "Logarithmique" else df['Close'].iloc[-1]
        dist_sig = (curr_y - y_pred[-1]) / std_err

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Performance (CAGR)", f"{cagr:.2%}")
        c2.metric("Volatilité", f"{vol:.2%}")
        c3.metric("R² (Précision)", f"{model.score(X, y_val):.3f}")
        c4.metric("Position Sigma", f"{dist_sig:.2f} σ")

        # Objectifs Rapides
        st.markdown("---")
        st.subheader("🎯 Niveaux Stratégiques")
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Vente (+2σ)", f"{to_p(y_pred[-1] + 2*std_err):.2f}")
        o2.metric("Haut Canal (+1σ)", f"{to_p(y_pred[-1] + std_err):.2f}")
        o3.metric("Bas Canal (-1σ)", f"{to_p(y_pred[-1] - std_err):.2f}")
        o4.metric("Achat (-2σ)", f"{to_p(y_pred[-1] - 2*std_err):.2f}")

    else:
        st.warning("Données insuffisantes ou ticker invalide.")
