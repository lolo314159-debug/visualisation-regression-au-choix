import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Analyse Quantitative Log", layout="wide")

st.title("📈 Analyse de Régression Log-Linéaire")
st.write("Analyse des tendances de long terme depuis 2000.")

# Configuration dans la barre latérale
uploaded_file = st.sidebar.file_uploader("Charger votre fichier Excel (colonne 'Ticker')", type="xlsx")
regression_type = st.sidebar.radio("Type de régression", ("Logarithmique (Recommandé)", "Linéaire"))
if uploaded_file:
    df_tickers = pd.read_excel(uploaded_file)
    
    # On récupère tous les noms de colonnes du fichier Excel
    columns = df_tickers.columns.tolist()
    
    # On demande à l'utilisateur quelle colonne contient les Tickers
    ticker_col = st.sidebar.selectbox("Sélectionnez la colonne des Tickers", columns)
    
    ticker_list = df_tickers[ticker_col].tolist()
    
    selected_ticker = st.sidebar.selectbox("Sélectionnez une action", ticker_list)
    
    if selected_ticker:
        # Téléchargement (yfinance gère bien les données multi-index, on nettoie ici)
        data = yf.download(selected_ticker, start="2000-01-01")
        
        if not data.empty:
            # On s'assure d'avoir un format plat (pour les téléchargements récents de yfinance)
            df = data[['Close']].copy()
            df.columns = ['Close'] 
            df.reset_index(inplace=True)
            
            # Préparation des données pour la régression
            df['Time_Index'] = np.arange(len(df)).reshape(-1, 1)
            X = df['Time_Index'].values.reshape(-1, 1)
            
            if regression_type == "Logarithmique (Recommandé)":
                # Calcul sur le Logarithme du prix
                y_raw = np.log(df['Close'].values)
                label_y = "Log(Prix)"
            else:
                # Calcul sur le prix brut
                y_raw = df['Close'].values
                label_y = "Prix"

            # 1. Ajustement du modèle
            model = LinearRegression()
            model.fit(X, y_raw)
            
            # 2. Prédictions et Sigma en espace Log (ou Linéaire)
            predictions_raw = model.predict(X)
            r_squared = model.score(X, y_raw)
            residuals = y_raw - predictions_raw
            std_dev = np.std(residuals)
            
            # Création des bandes
            if regression_type == "Logarithmique (Recommandé)":
                # On repasse en mode exponentiel pour l'affichage sur le prix réel
                df['Trend'] = np.exp(predictions_raw)
                df['Upper_1s'] = np.exp(predictions_raw + std_dev)
                df['Lower_1s'] = np.exp(predictions_raw - std_dev)
                df['Upper_2s'] = np.exp(predictions_raw + 2 * std_dev)
                df['Lower_2s'] = np.exp(predictions_raw - 2 * std_dev)
            else:
                df['Trend'] = predictions_raw
                df['Upper_1s'] = predictions_raw + std_dev
                df['Lower_1s'] = predictions_raw - std_dev
                df['Upper_2s'] = predictions_raw + 2 * std_dev
                df['Lower_2s'] = predictions_raw - 2 * std_dev

            # 3. Graphique Interactif
            fig = go.Figure()
            
            # Bandes de confiance (Sigma 2)
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper_2s'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower_2s'], fill='tonexty', 
                                     fillcolor='rgba(100, 100, 100, 0.2)', line=dict(width=0), name="Zone ±2σ"))
            
            # Prix et Tendance
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Prix", line=dict(color='#1f77b4', width=1.5)))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Trend'], name="Tendance Centrale", line=dict(color='orange', dash='dot')))

            fig.update_layout(
                title=f"Analyse {regression_type} pour {selected_ticker}",
                yaxis_type="log" if regression_type == "Logarithmique (Recommandé)" else "linear",
                template="plotly_dark",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            # 4. Indicateurs Clés
            current_price = df['Close'].iloc[-1]
            last_trend = df['Trend'].iloc[-1]
            # Calcul de la distance sigma actuelle
            current_y_raw = np.log(current_price) if regression_type == "Logarithmique (Recommandé)" else current_price
            last_pred_raw = predictions_raw[-1]
            dist_sigma = (current_y_raw - last_pred_raw) / std_dev

            c1, c2, c3 = st.columns(3)
            c1.metric("Prix Actuel", f"{current_price:.2f} $")
            c2.metric("R² (Précision)", f"{r_squared:.4f}")
            c3.metric("Position / Tendance", f"{dist_sigma:.2f} σ", 
                      delta_color="inverse" if dist_sigma > 2 else "normal")

            # 5. Interprétation
            st.subheader("💡 Verdict de l'analyse")
            if dist_sigma > 2:
                st.warning(f"**Surévaluation critique :** Le titre est à {dist_sigma:.1f} écarts-types au-dessus de sa moyenne historique. Historiquement, un retour vers la tendance est probable.")
            elif dist_sigma < -2:
                st.success(f"**Sous-évaluation majeure :** Le titre est à {abs(dist_sigma):.1f} écarts-types sous sa tendance. Cela a souvent constitué un point d'entrée historique.")
            else:
                st.info("Le titre oscille actuellement dans son corridor normal de croissance.")
