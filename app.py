import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Analyse Quantitative Pro", layout="wide")

st.title("📊 Analyse de Régression & Signaux Statistiques")
st.write("Calcul des tendances, bandes 1-σ / 2-σ, CAGR et volatilité depuis 2000.")

# --- BARRE LATÉRALE ---
uploaded_file = st.sidebar.file_uploader("1. Charger le fichier Excel", type="xlsx")

if uploaded_file:
    df_tickers = pd.read_excel(uploaded_file)
    columns = df_tickers.columns.tolist()
    
    # On cherche 'Ticker' ou on prend la 1ère colonne
    ticker_col = st.sidebar.selectbox("2. Colonne des Tickers", columns, 
                                     index=columns.index('Ticker') if 'Ticker' in columns else 0)
    
    ticker_list = df_tickers[ticker_col].dropna().unique().tolist()
    selected_ticker = st.sidebar.selectbox("3. Sélectionner l'action", ticker_list)
    
    regression_type = st.sidebar.radio("4. Modèle", ("Logarithmique", "Linéaire"))

    if selected_ticker:
        # --- RÉCUPÉRATION DES DONNÉES ---
        data = yf.download(selected_ticker, start="2000-01-01")
        
        if not data.empty:
            # Nettoyage pour les nouvelles versions de yfinance
            df = data[['Close']].copy()
            df.columns = ['Close']
            df.reset_index(inplace=True)
            
            # --- CALCULS FINANCIERS ---
            # Volatilité annualisée
            df['Returns'] = df['Close'].pct_change()
            volatility = df['Returns'].std() * np.sqrt(252)
            
            # CAGR
            days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
            cagr = (df['Close'].iloc[-1] / df['Close'].iloc[0])**(365.25 / days) - 1
            
            # --- RÉGRESSION ---
            df['Index'] = np.arange(len(df)).reshape(-1, 1)
            X = df['Index'].values.reshape(-1, 1)
            y_val = np.log(df['Close'].values) if regression_type == "Logarithmique" else df['Close'].values

            model = LinearRegression().fit(X, y_val)
            y_pred = model.predict(X)
            r2 = model.score(X, y_val)
            std_err = np.std(y_val - y_pred)
            
            # Fonction pour repasser en prix réel
            def to_price(v): return np.exp(v) if regression_type == "Logarithmique" else v

            df['Trend'] = to_price(y_pred)
            df['Up2'] = to_price(y_pred + 2 * std_err)
            df['Up1'] = to_price(y_pred + std_err)
            df['Low1'] = to_price(y_pred - std_err)
            df['Low2'] = to_price(y_pred - 2 * std_err)

            # --- GRAPHIQUE ---
            fig = go.Figure()
            # Bandes Sigma
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Up2'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Low2'], fill='tonexty', fillcolor='rgba(255,0,0,0.05)', name="Zone ±2σ"))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Up1'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Low1'], fill='tonexty', fillcolor='rgba(0,255,0,0.1)', name="Zone ±1σ"))
            
            # Prix et Tendance
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Prix", line=dict(color='white', width=1.2)))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Trend'], name="Tendance", line=dict(color='orange', dash='dash')))

            fig.update_layout(template="plotly_dark", height=600, 
                              yaxis_type="log" if regression_type == "Logarithmique" else "linear")
            st.plotly_chart(fig, use_container_width=True)

            # --- MÉTRIQUES ET OBJECTIFS ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CAGR", f"{cagr:.2%}")
            c2.metric("Volatilité", f"{volatility:.2%}")
            c3.metric("R²", f"{r2:.3f}")
            
            # Distance sigma actuelle
            curr_y = np.log(df['Close'].iloc[-1]) if regression_type == "Logarithmique" else df['Close'].iloc[-1]
            dist_sig = (curr_y - y_pred[-1]) / std_err
            c4.metric("Position Sigma", f"{dist_sig:.2f} σ")

            st.subheader("🎯 Objectifs et Supports Théoriques")
            t1, t2 = st.columns(2)
            with t1:
                st.write(f"Vente (+2σ) : **{df['Up2'].iloc[-1]:.2f}**")
                st.write(f"Objectif (+1σ) : **{df['Up1'].iloc[-1]:.2f}**")
            with t2:
                st.write(f"Support (-1σ) : **{df['Low1'].iloc[-1]:.2f}**")
                st.write(f"Achat (-2σ) : **{df['Low2'].iloc[-1]:.2f}**")

            # --- ANALYSE FINALE ---
            st.divider()
            st.subheader("📝 Analyse de la situation")
            
            eval_pos = "surévalué" if dist_sig > 1 else "sous-évalué" if dist_sig < -1 else "à sa juste valeur"
            
            st.info(f"""
            **Analyse Long Terme :** Depuis 2000, le titre affiche une croissance annuelle (**CAGR**) de **{cagr:.2%}**. 
            La tendance est très **{'fiable' if r2 > 0.8 else 'volatile'}** (R² de {r2:.2f}).
            
            **Position Actuelle :** Avec une position de **{dist_sig:.2f} σ**, le titre est actuellement **{eval_pos}** par rapport à son couloir historique. 
            Une position au-delà de ±2σ est statistiquement rare et indique souvent un point de retournement majeur.
            """)
