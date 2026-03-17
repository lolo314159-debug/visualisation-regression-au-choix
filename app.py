import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Analyse Quantitative Avancée", layout="wide")

st.title("📊 Analyse de Régression & Signaux Statistiques")
st.write("Analyse de la tendance long terme, volatilité et objectifs théoriques.")

# --- SIDEBAR : CHARGEMENT ET CONFIGURATION ---
uploaded_file = st.sidebar.file_uploader("1. Charger le fichier Excel", type="xlsx")

if uploaded_file:
    df_tickers = pd.read_excel(uploaded_file)
    columns = df_tickers.columns.tolist()
    
    # Correction automatique ou manuelle de la colonne
    default_index = columns.index('Ticker') if 'Ticker' in columns else 0
    ticker_col = st.sidebar.selectbox("2. Colonne des Tickers", columns, index=default_index)
    
    ticker_list = df_tickers[ticker_col].dropna().unique().tolist()
    selected_ticker = st.sidebar.selectbox("3. Sélectionner l'action", ticker_list)
    
    regression_type = st.sidebar.radio("4. Type de modèle", ("Logarithmique (Recommandé)", "Linéaire"))

    if selected_ticker:
        # --- DATA FETCHING ---
        data = yf.download(selected_ticker, start="2000-01-01")
        
        if not data.empty:
            df = data[['Close']].copy()
            df.columns = ['Close']
            df.reset_index(inplace=True)
            
            # --- CALCULS FINANCIERS ---
            # Volatilité annualisée (basée sur les rendements logarithmiques)
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            volatility = df['Log_Ret'].std() * np.sqrt(252)
            
            # CAGR (Taux de croissance annuel composé)
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            num_years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
            cagr = (end_price / start_price)**(1 / num_years) - 1
            
            # --- RÉGRESSION ---
            df['Time_Index'] = np.arange(len(df)).reshape(-1, 1)
            X = df['Time_Index'].values.reshape(-1, 1)
            
            if regression_type == "Logarithmique (Recommandé)":
                y_raw = np.log(df['Close'].values)
            else:
                y_raw = df['Close'].values

            model = LinearRegression()
            model.fit(X, y_raw)
            predictions_raw = model.predict(X)
            r_squared = model.score(X, y_raw)
            std_dev = np.std(y_raw - predictions_raw)
            
            # Génération des courbes
            def transform_back(val):
                return np.exp(val) if regression_type == "Logarithmique (Recommandé)" else val

            df['Trend'] = transform_back(predictions_raw)
            df['Up2'] = transform_back(predictions_raw + 2 * std_dev)
            df['Up1'] = transform_back(predictions_raw + std_dev)
            df['Low1'] = transform_back(predictions_raw - std_dev)
            df['Low2'] = transform_back(predictions_raw - 2 * std_dev)

            # --- VISUALISATION ---
            fig = go.Figure()
            # Zones Sigma
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Up2'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Low2'], fill='tonexty', fillcolor='rgba(255, 0, 0, 0.05)', line=dict(width=0), name="Zone ±2σ (Extrême)"))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Up1'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Low1'], fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', line=dict(width=0), name="Zone ±1σ (Standard)"))
            
            # Prix et Tendance
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Prix", line=dict(color='white', width=1.5)))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Trend'], name="Tendance", line=dict(color='orange', dash='dot')))

            fig.update_layout(template="plotly_dark", height=600, yaxis_type="log" if regression_type == "Logarithmique (Recommandé)" else "linear")
            st.plotly_chart(fig, use_container_width=True)

            # --- TABLEAU DE BORD DES MÉTRIQUES ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("CAGR (Croissance)", f"{cagr:.2%}")
            col2.metric("Volatilité (Ann.)", f"{volatility:.2%}")
            col3.metric("R² (Fidélité)", f"{r_squared:.3f}")
            
            current_price = df['Close'].iloc[-1]
            last_pred = predictions_raw[-1]
            current_raw = np.log(current_price) if regression_type == "Logarithmique (Recommandé)" else current_price
            dist_sigma = (current_raw - last_pred) / std_dev
            col4.metric("Position Sigma", f"{dist_sigma:.2f} σ")

            # --- PRIX SUPPORTS ET OBJECTIFS ---
            st.subheader("🎯 Objectifs et Supports Théoriques (Prix actuels)")
            t1, t2 = st.columns(2)
            with t1:
                st.write("**Objectifs (Haut de canal)**")
                st.write(f"Vente Extrême (+2σ) : **{df['Up2'].iloc[-1]:.2f}**")
                st.write(f"Objectif Standard (+1σ) : **{df['Up1'].iloc[-1]:.2f}**")
            with t2:
                st.write("**Supports (Bas de canal)**")
                st.write(f"Support Standard (-1σ) : **{df['Low1'].iloc[-1]:.2f}**")
                st.write(f"Achat Extrême (-2σ) : **{df['Low2'].iloc[-1]:.2f}**")

            # --- COMMENTAIRE PRÉCIS ---
            st.divider()
            st.subheader("📝 Analyse de la situation")
            
            # Analyse Tendance
            tendance_txt = "robuste" if r_squared > 0.8 else "modérée"
            croissance_txt = "exceptionnelle" if cagr > 0.15 else "stable"
            
            # Analyse Position
            if dist_sigma > 1.5:
                pos_txt = "en zone de surchauffe. Le prix est significativement éloigné de sa moyenne historique, suggérant une prudence à court terme."
            elif dist_sigma < -1.5:
                pos_txt = "en zone de sous-évaluation historique. Statistiquement, le titre offre un point d'entrée attractif par rapport à sa tendance."
            else:
                pos_txt = "proche de sa valeur d'équilibre. Le marché valorise le titre en ligne avec ses fondamentaux historiques."

            st.info(f"""
            **Évolution Long Terme :** Depuis 2000, **{selected_ticker}** présente une croissance annuelle composée (CAGR) de **{cagr:.2%** avec une tendance jugée **{tendance_txt}** (R² de {r_squared:.2f}). 
            La volatilité de **{volatility:.2%}** indique un profil de risque {'élevé' if volatility > 0.3 else 'modéré'}.
            
            **Position Actuelle :** Le titre se situe actuellement à **{dist_sigma:.2f} écart-type(s)** de sa droite de régression. Il est donc **{pos_txt}**
            """)
