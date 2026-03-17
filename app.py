import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Analyse Quantitative Premium", layout="wide")

# --- CHARGEMENT DU FICHIER ---
uploaded_file = st.sidebar.file_uploader("1. Charger le fichier Excel", type="xlsx")

if uploaded_file:
    df_tickers = pd.read_excel(uploaded_file)
    columns = df_tickers.columns.tolist()
    
    ticker_col = st.sidebar.selectbox("Colonne des Tickers", columns, 
                                     index=columns.index('Ticker') if 'Ticker' in columns else 0)
    
    # Optionnel : chercher une colonne 'Nom' pour le titre
    name_col = next((c for c in columns if 'nom' in c.lower()), None)
    
    ticker_list = df_tickers[ticker_col].dropna().unique().tolist()
    selected_ticker = st.sidebar.selectbox("Sélectionner l'action", ticker_list)
    
    # Récupération du nom complet pour le titre
    display_name = ""
    if name_col:
        display_name = df_tickers[df_tickers[ticker_col] == selected_ticker][name_col].iloc[0]

    regression_type = st.sidebar.radio("Modèle de calcul", ("Logarithmique", "Linéaire"))

    if selected_ticker:
        data = yf.download(selected_ticker, start="2000-01-01")
        
        if not data.empty:
            df = data[['Close']].copy()
            df.columns = ['Close']
            df.reset_index(inplace=True)
            
            # --- CALCULS ---
            df['Returns'] = df['Close'].pct_change()
            volatility = df['Returns'].std() * np.sqrt(252)
            days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
            cagr = (df['Close'].iloc[-1] / df['Close'].iloc[0])**(365.25 / days) - 1
            
            df['Index'] = np.arange(len(df)).reshape(-1, 1)
            X = df['Index'].values.reshape(-1, 1)
            y_val = np.log(df['Close'].values) if regression_type == "Logarithmique" else df['Close'].values

            model = LinearRegression().fit(X, y_val)
            y_pred = model.predict(X)
            std_err = np.std(y_val - y_pred)
            
            def to_p(v): return np.exp(v) if regression_type == "Logarithmique" else v

            # --- GRAPHIQUE ÉPURÉ ---
            title_text = f"<b>{display_name}</b> ({selected_ticker})" if display_name else f"<b>Analyse : {selected_ticker}</b>"
            
            fig = go.Figure()

            # Bandes Sigma 2 (Très discrètes)
            fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred + 2*std_err), line=dict(color='rgba(255,100,100,0.2)', width=1, dash='dot'), name="±2σ (Extrême)"))
            fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred - 2*std_err), line=dict(color='rgba(255,100,100,0.2)', width=1, dash='dot'), showlegend=False))
            
            # Bandes Sigma 1 (Plus visibles mais légères)
            fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred + std_err), line=dict(color='rgba(100,255,100,0.3)', width=1), name="±1σ (Canal)"))
            fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred - std_err), line=dict(color='rgba(100,255,100,0.3)', width=1), showlegend=False))
            
            # Tendance Centrale
            fig.add_trace(go.Scatter(x=df['Date'], y=to_p(y_pred), line=dict(color='orange', width=1.5, dash='dash'), name="Tendance"))

            # PRIX (L'élément principal)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='white', width=2.5), name="Prix de Clôture"))

            fig.update_layout(
                title=dict(text=title_text, font=dict(size=24)),
                template="plotly_dark",
                hovermode="x unified",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Prix ($)"),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Année"),
                yaxis_type="log" if regression_type == "Logarithmique" else "linear"
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # --- RÉSUMÉ ET ANALYSE ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CAGR", f"{cagr:.2%}")
            c2.metric("Volatilité", f"{volatility:.2%}")
            c3.metric("R²", f"{model.score(X, y_val):.3f}")
            curr_y = np.log(df['Close'].iloc[-1]) if regression_type == "Logarithmique" else df['Close'].iloc[-1]
            dist_sig = (curr_y - y_pred[-1]) / std_err
            c4.metric("Position Sigma", f"{dist_sig:.2f} σ")

            st.markdown(f"### 🎯 Objectifs théoriques pour **{selected_ticker}**")
            o1, o2, o3, o4 = st.columns(4)
            o1.caption("Vente Extrême (+2σ)")
            o1.write(f"**{to_p(y_pred[-1] + 2*std_err):.2f}**")
            o2.caption("Objectif (+1σ)")
            o2.write(f"**{to_p(y_pred[-1] + std_err):.2f}**")
            o3.caption("Support (-1σ)")
            o3.write(f"**{to_p(y_pred[-1] - std_err):.2f}**")
            o4.caption("Achat Extrême (-2σ)")
            o4.write(f"**{to_p(y_pred[-1] - 2*std_err):.2f}**")

            st.divider()
            st.subheader("📝 Commentaire de situation")
            if dist_sig > 1.5:
                status, color = "SURÉVALUÉ", "orange"
            elif dist_sig < -1.5:
                status, color = "SOUS-ÉVALUÉ", "green"
            else:
                status, color = "DANS SA MOYENNE", "grey"

            st.markdown(f"""
            Le titre **{selected_ticker}** présente historiquement une croissance annuelle de **{cagr:.2%}**. 
            Actuellement, avec un écart de **{dist_sig:.2f} σ**, le cours est considéré comme **:{color}[{status}]**.
            
            * **Scénario de hausse** : Un retour vers le haut du canal (+1σ) impliquerait un cours de **{to_p(y_pred[-1] + std_err):.2f}**.
            * **Scénario de baisse** : Un repli sur le support (-1σ) situerait le prix autour de **{to_p(y_pred[-1] - std_err):.2f}**.
            """)
