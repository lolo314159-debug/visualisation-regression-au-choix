import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Analyse Quantitative", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
method = st.sidebar.radio("Source :", ("Saisie Manuelle", "Fichier Excel"))

selected_ticker = ""
name_display = ""

if method == "Fichier Excel":
    file = st.sidebar.file_uploader("Charger Excel", type="xlsx")
    if file:
        df_excel = pd.read_excel(file)
        cols = df_excel.columns.tolist()
        t_col = st.sidebar.selectbox("Colonne Tickers", cols)
        n_col = next((c for c in cols if "nom" in c.lower()), None)
        ticker_list = df_excel[t_col].dropna().unique().tolist()
        selected_ticker = st.sidebar.selectbox("Choisir l'action", ticker_list)
        if n_col:
            name_display = df_excel[df_excel[t_col] == selected_ticker][n_col].iloc[0]
else:
    selected_ticker = st.sidebar.text_input("Ticker (ex: NVDA, OR.PA)", "MSFT").upper()

reg_mode = st.sidebar.radio("Modèle :", ("Logarithmique", "Linéaire"))

# --- CALCULS ET GRAPHIQUE ---
if selected_ticker:
    ticker_obj = yf.Ticker(selected_ticker)
    data = ticker_obj.history(start="2000-01-01")
    
    if not name_display:
        try: name_display = ticker_obj.info.get('longName', selected_ticker)
        except: name_display = selected_ticker

    if not data.empty and len(data) > 30:
        df = data[['Close']].copy().dropna().reset_index()
        
        # Régression
        df['Idx'] = np.arange(len(df)).reshape(-1, 1)
        X = df['Idx'].values.reshape(-1, 1)
        y = np.log(df['Close'].values) if reg_mode == "Logarithmique" else df['Close'].values
        
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        std_dev = np.std(y - y_pred)
        
        def rev(v): return np.exp(v) if reg_mode == "Logarithmique" else v

        # --- PLOTLY (FOND BLANC) ---
        fig = go.Figure()

        # Zones Pastel
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred + 2*std_dev), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred - 2*std_dev), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.05)', line_color='rgba(0,0,0,0)', name="±2σ"))
        
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred + std_dev), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred - std_dev), fill='tonexty', fillcolor='rgba(0, 200, 0, 0.1)', line_color='rgba(0,0,0,0)', name="±1σ"))

        # Tendance et Prix
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred), line=dict(color='orange', width=1, dash='dash'), name="Tendance"))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='black', width=2), name="Prix"))

        # Mise en page simplifiée pour éviter le ValueError
        fig.update_layout(
            title=f"<b>{name_display}</b> | {selected_ticker}",
            template="plotly_white",
            paper_bgcolor='white',
            plot_bgcolor='white',
            yaxis_type="log" if reg_mode == "Logarithmique" else "linear",
            yaxis=dict(side="right", gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(showgrid=False),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- METRICS ---
        c1, c2, c3 = st.columns(3)
        # CAGR
        years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
        cagr = (df['Close'].iloc[-1] / df['Close'].iloc[0])**(1/years) - 1
        # Sigma
        cur_y = np.log(df['Close'].iloc[-1]) if reg_mode == "Logarithmique" else df['Close'].iloc[-1]
        sig_pos = (cur_y - y_pred[-1]) / std_dev
        
        c1.metric("Performance (CAGR)", f"{cagr:.2%}")
        c2.metric("Fiabilité (R²)", f"{model.score(X, y):.3f}")
        c3.metric("Position Sigma", f"{sig_pos:.2f} σ")
        
        st.write(f"**Objectif (+1σ) :** {rev(y_pred[-1] + std_dev):.2f} | **Support (-1σ) :** {rev(y_pred[-1] - std_dev):.2f}")
    else:
        st.error("Données introuvables ou ticker incorrect.")
