import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(page_title="Analyse Quantitative Pro", layout="wide")

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
    selected_ticker = st.sidebar.text_input("Ticker (ex: TRN.MI, NVDA)", "MSFT").upper()

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
        
        # Volatilité Annualisée
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        volatility = df['Log_Ret'].std() * np.sqrt(252)
        
        # Régression
        df['Idx'] = np.arange(len(df)).reshape(-1, 1)
        X = df['Idx'].values.reshape(-1, 1)
        y = np.log(df['Close'].values) if reg_mode == "Logarithmique" else df['Close'].values
        
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = model.score(X, y)
        std_dev = np.std(y - y_pred)
        
        def rev(v): return np.exp(v) if reg_mode == "Logarithmique" else v

        # --- PLOTLY (FOND BLANC) ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred + 2*std_dev), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred - 2*std_dev), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.05)', line_color='rgba(0,0,0,0)', name="±2σ"))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred + std_dev), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred - std_dev), fill='tonexty', fillcolor='rgba(0, 200, 0, 0.1)', line_color='rgba(0,0,0,0)', name="±1σ"))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred), line=dict(color='orange', width=1, dash='dash'), name="Tendance"))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='black', width=2.5), name="Prix"))

        fig.update_layout(
            title=f"<b>{name_display}</b> | {selected_ticker}",
            template="plotly_white", paper_bgcolor='white', plot_bgcolor='white',
            yaxis_type="log" if reg_mode == "Logarithmique" else "linear",
            yaxis=dict(side="right", gridcolor='rgba(0,0,0,0.1)'), xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- METRICS ---
        years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
        cagr = (df['Close'].iloc[-1] / df['Close'].iloc[0])**(1/years) - 1
        cur_y = np.log(df['Close'].iloc[-1]) if reg_mode == "Logarithmique" else df['Close'].iloc[-1]
        sig_pos = (cur_y - y_pred[-1]) / std_dev
        
        # --- CALCUL DU SCORE DE QUALITÉ (SQQ) ---
        # R2 (0-4 pts) | CAGR (0-4 pts) | Volatilité (0-2 pts)
        score_r2 = r2 * 4
        score_cagr = min(max(cagr * 20, 0), 4) # 4 pts si CAGR >= 20%
        score_vol = max(2 - (volatility * 2), 0) # 2 pts si vol <= 10%, 0 si >= 100%
        total_score = score_r2 + score_cagr + score_vol

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", f"{cagr:.2%}")
        c2.metric("Volatilité", f"{volatility:.2%}")
        c3.metric("R² (Fiabilité)", f"{r2:.3f}")
        c4.metric("SCORE QUALITÉ", f"{total_score:.1f} / 10")

        # --- NIVEAUX ---
        st.subheader("🎯 Niveaux Statistiques (Prix actuels)")
        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("Vente (+2σ)", f"{rev(y_pred[-1] + 2*std_dev):.2f}")
        o2.metric("Cible (+1σ)", f"{rev(y_pred[-1] + std_dev):.2f}")
        o3.metric("Moyenne", f"{rev(y_pred[-1]):.2f}")
        o4.metric("Support (-1σ)", f"{rev(y_pred[-1] - std_dev):.2f}")
        o5.metric("Achat (-2σ)", f"{rev(y_pred[-1] - 2*std_dev):.2f}")

        # --- DIAGNOSTIC ---
        st.divider()
        st.subheader("📝 Diagnostic Quantitatif")
        
        d1, d2, d3 = st.columns(3)
        with d1:
            st.write(f"**Analyse CAGR ({cagr:.1%})**")
            if cagr > 0.15: st.write("Performance supérieure. Profil à haut rendement composé.")
            elif cagr > 0.07: st.write("Performance normative. Croissance en ligne avec les indices actions.")
            else: st.write("Performance sous-optimale. Rendement inférieur à la moyenne historique des actions.")

        with d2:
            st.write(f"**Analyse Volatilité ({volatility:.1%})**")
            if volatility > 0.40: st.write("Instabilité structurelle. Risque de perte en capital élevé sur courte période.")
            elif volatility > 0.20: st.write("Nervosité standard. Volatilité typique d'un actif risqué.")
            else: st.write("Stabilité défensive. Faible sensibilité aux bruits de marché.")

        with d3:
            st.write(f"**Analyse R² ({r2:.2f})**")
            if r2 > 0.90: st.write("Corrélation temporelle extrême. Le modèle est une référence fiable.")
            elif r2 > 0.70: st.write("Tendance identifiable. Les bandes sigma sont pertinentes.")
            else: st.write("Absence de tendance claire. Les calculs sigma sont peu prédictifs.")

        # Verdict
        st.write(f"**SYNTHÈSE TECHNIQUE :**")
        if total_score > 8: st.write("⭐️ **Actif Premium :** Tendance historique saine, performante et régulière.")
        elif total_score > 5: st.write("⚖️ **Actif Standard :** Profil équilibré avec des phases cycliques marquées.")
        else: st.write("⚠️ **Actif Spéculatif/Dégradé :** Faible prédictibilité ou performance médiocre.")

        # Position actuelle
        if abs(sig_pos) > 2:
            st.error(f"ALERTE : Écart-type de {sig_pos:.2f}σ. Rupture de canal détectée.")
        elif abs(sig_pos) > 1:
            st.warning(f"DÉVIATION : Écart de {sig_pos:.2f}σ. Le titre s'éloigne de son équilibre historique.")
        else:
            st.success(f"ZONE NEUTRE : Écart de {sig_pos:.2f}σ. Prix en adéquation avec la tendance long terme.")

    else:
        st.error("Données historiques insuffisantes ou ticker invalide.")
