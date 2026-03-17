import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Analyse Quantitative Premium (Light)", layout="wide")

# --- BARRE LATÉRALE ---
st.sidebar.header("Configuration")
input_method = st.sidebar.radio("Choisir la méthode :", ("Fichier Excel", "Saisie Manuelle"))

selected_ticker = ""
display_name = ""

if input_method == "Fichier Excel":
    uploaded_file = st.sidebar.file_uploader("1. Charger Excel", type="xlsx")
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
    # Par défaut sur MSFT pour l'exemple
    selected_ticker = st.sidebar.text_input("2. Saisir Ticker (ex: AAPL, AIR.PA)", "MSFT").upper()

regression_type = st.sidebar.radio("3. Modèle de calcul", ("Logarithmique", "Linéaire"))

# --- RÉCUPÉRATION ET CALCULS ---
if selected_ticker:
    with st.spinner(f'Téléchargement et analyse de {selected_ticker}...'):
        ticker_obj = yf.Ticker(selected_ticker)
        # Télécharger depuis 2000
        data = ticker_obj.history(start="2000-01-01")
        
        # Récupérer le nom complet via Yahoo Finance si pas trouvé dans l'Excel
        if not display_name:
            try: display_name = ticker_obj.info.get('longName', selected_ticker)
            except: display_name = selected_ticker

    if not data.empty and len(data) > 20:
        # Nettoyage
        df = data[['Close']].copy()
        df = df.dropna().reset_index()
        
        # Maths pour la régression
        df['Index'] = np.arange(len(df)).reshape(-1, 1)
        X = df['Index'].values.reshape(-1, 1)
        # Régression sur log ou prix brut
        y_val = np.log(df['Close'].values) if regression_type == "Logarithmique" else df['Close'].values
        
        model = LinearRegression().fit(X, y_val)
        y_pred = model.predict(X)
        std_err = np.std(y_val - y_pred)
        
        # Fonction utilitaire pour repasser en prix réel
        def to_price(v): return np.exp(v) if regression_type == "Logarithmique" else v

        # --- GRAPHIQUE ÉPURÉ (FONDS BLANC) ---
        title_text = f"<b>{display_name}</b> <span style='color:#555;'>| {selected_ticker}</span>"
        
        fig = go.Figure()

        # Zones de volatilité Pastel (Arrière-plan)
        # Zone 2-Sigma (Rouge très clair)
        fig.add_trace(go.Scatter(x=df['Date'], y=to_price(y_pred + 2*std_err), line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=to_price(y_pred - 2*std_err), fill='tonexty', 
                                 fillcolor='rgba(255, 100, 100, 0.08)', line=dict(width=0), name="Zone ±2σ (Extrême)"))

        # Zone 1-Sigma (Vert très clair)
        fig.add_trace(go.Scatter(x=df['Date'], y=to_price(y_pred + std_err), line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=to_price(y_pred - std_err), fill='tonexty', 
                                 fillcolor='rgba(100, 255, 150, 0.15)', line=dict(width=0), name="Zone ±1σ (Canal)"))

        # Lignes de tendance
        fig.add_trace(go.Scatter(x=df['Date'], y=to_price(y_pred), line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5, dash='dash'), name="Tendance"))

        # PRIX (Ligne principale, Noire sur fond blanc)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='#000000', width=2.5), name="Prix de Clôture"))

        # Mise en page (Theme Light)
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=26, color="black")),
            template="plotly_white", # Force le thème clair
            hovermode="x unified",
            yaxis_type="log" if regression_type == "Logarithmique" else "linear",
            paper_bgcolor='white', # Fond du papier blanc
            plot_bgcolor='white',  # Fond du graphique blanc
            yaxis=dict(
                gridcolor='rgba(0,0,0,0.05)', # Grille très légère
                side="right", 
                title="Prix ($)",
                tickfont=dict(color="#333"),
                titlefont=dict(color="#333")
            ),
            xaxis=dict(
                showgrid=False,
                title="Année",
                tickfont=dict(color="#333"),
                titlefont=dict(color="#333")
            ),
            legend=dict(font=dict(color="#333"), bgcolor='rgba(255,255,255,0.7)')
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- TABLEAU DE BORD (KPIs) ---
        # Calculs KPIs
        days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
        cagr = (df['Close'].iloc[-1] / df['Close'].iloc[0])**(365.25 / days) - 1
        # Volatilité annualisée basée sur les rendements logarithmiques
        vol = (np.
