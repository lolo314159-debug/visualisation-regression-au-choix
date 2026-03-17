import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Analyse Quantitative Premium", layout="wide")

# --- CHARGEMENT DU FICHIER ---
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("1. Charger le fichier Excel", type="xlsx")

if uploaded_file:
    df_tickers = pd.read_excel(uploaded_file)
    columns = df_tickers.columns.tolist()
    
    ticker_col = st.sidebar.selectbox("Colonne des Tickers", columns, 
                                     index=columns.index('Ticker') if 'Ticker' in columns else 0)
    
    # Recherche du nom de l'entreprise pour le titre
    name_col = next((c for c in columns if "nom" in c.lower()), None)
    
    ticker_list = df_tickers[ticker_col].dropna().unique().tolist()
    selected_ticker = st.sidebar.selectbox("Sélectionner l'action", ticker_list)
    
    display_name = ""
    if name_col:
        display_name = df_tickers[df_tickers[ticker_col] == selected_ticker][name_col].iloc[0]

    regression_type = st.sidebar.radio("Modèle de calcul", ("Logarithmique", "Linéaire"))

    if selected_ticker:
        # Téléchargement avec gestion d'erreur
        with st.spinner('Téléchargement des données...'):
            data = yf.download(selected_ticker, start="2000-01-01")
        
        if not data.empty and len(data) > 10:
            # Nettoyage des données (important pour éviter le crash scikit-learn)
            df = data[['Close']].copy()
            df.columns = ['Close']
            df = df.dropna() 
            df.reset_index(inplace=True)
            
            # --- CALCULS FINANCIERS ---
            days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
            cagr = (df['Close'].iloc[-1] / df['Close'].iloc[0])**(365.25 / days) - 1
            volatility = (np.log(df['Close'] / df['Close'].shift(1))).std() * np.sqrt(252)
            
            # --- RÉGRESSION ---
            df['Index'] = np.arange(len(df)).reshape(-1, 1)
            X = df['Index'].values.reshape(-1, 1)
            y_val = np.log(df['Close'].values) if regression_type == "Logarithmique" else df['Close'].values

            # Sécurité : On vérifie qu'il n'y a pas d'infinis ou de NaN
            mask = np.isfinite(y_val)
            X, y_val = X[mask], y_val[mask]

            model = LinearRegression().fit(X, y_val)
            y_pred = model.predict(X)
            std_err = np.std(y_val - y_pred)
            
            def to_p(v): return np.exp(v) if regression_type == "Logarithmique" else v

            # --- GRAPHIQUE ---
            title_text = f"<b>{display_name}</b> ({selected_ticker})" if display_name else f"<b>{selected_ticker}</b>"
            
            fig = go.Figure()

            # Bandes Sigma (Arrière-plan)
            fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred + 2*std_err), line=dict(color='rgba(255,100,100,0.15)', width=1, dash='dot'), name="±2σ (Extrême)"))
            fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred - 2*std_err), line=dict(color='rgba(255,100,100,0.15)', width=1, dash='dot'), showlegend=False))
            
            fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred + std_err), line=dict(color='rgba(100,255,100,0.2)', width=1), name="±1σ (Canal)"))
            fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred - std_err), line=dict(color='rgba(100,255,100,0.2)', width=1), showlegend=False))
            
            # Tendance
            fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred), line=dict(color='rgba(255,165,0,0.5)', width=1.5, dash='dash'), name="Tendance"))

            # PRIX (Mis en avant)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='#FFFFFF', width=2.5), name="Prix de Clôture"))

            fig.update_layout(
                title=dict(text=title_text, font=dict(size=26, color="white")),
                template="plotly_dark",
                hovermode="x unified",
                xaxis=dict(showgrid=False, title="Année"),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Prix ($)", side="right"),
                yaxis_type="log" if regression_type == "Logarithmique" else "linear",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # --- KPI & ANALYSE ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Performance (CAGR)", f"{cagr:.2%}")
            c2.metric("Volatilité", f"{volatility:.2%}")
            c3.metric("Fiabilité (R²)", f"{model.score(X, y_val):.3f}")
            
            curr_y = np.log(df['Close'].iloc[-1]) if regression_type == "Logarithmique" else df['Close'].iloc[-1]
            dist_sig = (curr_y - y_pred[-1]) / std_err
            
            color_sigma = "inverse" if dist_sig > 1.5 else "normal"
            c4.metric("Position Sigma", f"{dist_sig:.2f} σ", delta_color=color_sigma)

            st.markdown("---")
            # Objectifs
            st.subheader("🎯 Objectifs et Supports")
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Vente (+2σ)", f"{to_p(y_pred[-1] + 2*std_err):.2f}")
            o2.metric("Cible (+1σ)", f"{to_p(y_pred[-1] + std_err):.2f}")
            o3.metric("Support (-1σ)", f"{to_p(y_pred[-1] - std_err):.2f}")
            o4.metric("Achat (-2σ)", f"{to_p(y_pred[-1] - 2*std_err):.2f}")

            # Commentaire
            st.divider()
            status = "SURÉVALUÉ" if dist_sig > 1 else "SOUS-ÉVALUÉ" if dist_sig < -1 else "À SA VALEUR"
            st.info(f"**Analyse :** Le titre est actuellement **{status}**. Sa croissance historique de **{cagr:.2%}** par an reste le moteur principal du cours.")
        else:
            st.error(f"Impossible de récupérer des données pour le ticker '{selected_ticker}'. Vérifiez l'orthographe sur Yahoo Finance.")
