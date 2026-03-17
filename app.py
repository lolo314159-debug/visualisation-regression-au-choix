import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from fpdf import FPDF
import io

st.set_page_config(page_title="Analyse Quantitative", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
method = st.sidebar.radio("Source :", ("Saisie Manuelle", "Fichier Excel (ticker+nom)"))

selected_ticker = ""
name_display = ""

if method == "Fichier Excel (ticker+nom)":
    file = st.sidebar.file_uploader("Charger Excel", type="xlsx")
    if file:
        df_excel = pd.read_excel(file)
        cols = df_excel.columns.tolist()
        t_col = st.sidebar.selectbox("Colonne Tickers", cols)
        n_col = next((c for c in cols if "nom" in c.lower()), None)
        ticker_list = df_excel[t_col].dropna().unique().tolist()
        selected_ticker = st.sidebar.selectbox("Choisir l'action", ticker_list)
        if n_col:
            name_display = str(df_excel[df_excel[t_col] == selected_ticker][n_col].iloc[0])
else:
    selected_ticker = st.sidebar.text_input("Ticker (ex: NVDA, OR.PA)", "MSFT").upper()

reg_mode = st.sidebar.radio("Modèle :", ("Logarithmique", "Linéaire"))

# --- LOGIQUE D'ANALYSE ---
if selected_ticker:
    ticker_obj = yf.Ticker(selected_ticker)
    data = ticker_obj.history(start="2000-01-01")
    
    if not name_display:
        try: name_display = ticker_obj.info.get('longName', selected_ticker)
        except: name_display = selected_ticker

    if not data.empty and len(data) > 30:
        df = data[['Close']].copy().dropna().reset_index()
        
        # Préparation des données (sécurité scalaires)
        df['Idx'] = np.arange(len(df)).reshape(-1, 1)
        X = df['Idx'].values.reshape(-1, 1)
        y = np.log(df['Close'].values) if reg_mode == "Logarithmique" else df['Close'].values
        
        # Régression
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X).flatten()
        r2 = float(model.score(X, y))
        std_dev = float(np.std(y - y_pred))
        
        def rev(v): return np.exp(v) if reg_mode == "Logarithmique" else v

        # --- GRAPH ET AFFICHAGE ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred + 2*std_dev), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred - 2*std_dev), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.05)', line_color='rgba(0,0,0,0)', name="Zone ±2σ"))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred + std_dev), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred - std_dev), fill='tonexty', fillcolor='rgba(0, 200, 0, 0.1)', line_color='rgba(0,0,0,0)', name="Zone ±1σ"))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred), line=dict(color='orange', width=1, dash='dash'), name="Tendance"))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='black', width=2), name="Prix"))

        fig.update_layout(
            title=f"<b>{name_display}</b> | {selected_ticker}",
            template="plotly_white", paper_bgcolor='white', plot_bgcolor='white',
            yaxis_type="log" if reg_mode == "Logarithmique" else "linear",
            yaxis=dict(side="right", gridcolor='rgba(0,0,0,0.1)'), xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- CALCULS MÉTRIQUES FINAUX ---
        last_price = float(df['Close'].iloc[-1])
        years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
        cagr = float((last_price / df['Close'].iloc[0])**(1/years) - 1)
        log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        vol = float(log_ret.std() * np.sqrt(252))
        
        cur_y_val = np.log(last_price) if reg_mode == "Logarithmique" else last_price
        sig_pos = float((cur_y_val - y_pred[-1]) / std_dev)

        # SCORE DE QUALITÉ (Sur 10)
        score_q = (r2 * 5) + (np.clip(cagr * 20, 0, 3)) + (np.clip(2 - vol, 0, 2))
        
        # --- BLOC OBJECTIFS (Ton affichage préféré) ---
        st.subheader("🎯 Objectifs et Supports Théoriques")
        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("Vente (+2σ)", f"{rev(y_pred[-1] + 2*std_dev):.2f}")
        o2.metric("Objectif (+1σ)", f"{rev(y_pred[-1] + std_dev):.2f}")
        o3.metric("Moyenne", f"{rev(y_pred[-1]):.2f}")
        o4.metric("Support (-1σ)", f"{rev(y_pred[-1] - std_dev):.2f}")
        o5.metric("Achat (-2σ)", f"{rev(y_pred[-1] - 2*std_dev):.2f}")

        # --- ANALYSE FINALE ENRICHIE ---
        st.divider()
        c_diag1, c_diag2 = st.columns([2, 1])
        
        with c_diag1:
            st.subheader("📝 Diagnostic de l'Investisseur")
            # Logique d'analyse plus poussée
            if r2 > 0.9: 
                qualite_msg = "Exceptionnelle (Trajectoire quasi-parfaite)"
            elif r2 > 0.7:
                qualite_msg = "Robuste (Croissance structurée)"
            else:
                qualite_msg = "Cyclique ou Instable"
                
            st.write(f"**Qualité de la tendance :** {qualite_msg} (R²: {r2:.4f})")
            st.write(f"**Performance annuelle moy. (CAGR) :** {cagr:.2%}")
            st.write(f"**Score de Qualité Global :** {score_q:.1f} / 10")
            
            # Avis contextuel
            if sig_pos < -1.5:
                st.success(f"🔥 **OPPORTUNITÉ :** Le titre subit une décote rare ({sig_pos:.2f}σ). Probabilité de retour à la moyenne élevée.")
            elif sig_pos > 1.5:
                st.error(f"🚨 **PRUDENCE :** Surchauffe statistique (+{sig_pos:.2f}σ). Le risque de correction est maximal.")
            else:
                st.info(f"⚖️ **NEUTRE :** Le prix est en phase avec son corridor historique ({sig_pos:.2f}σ).")

        with c_diag2:
            st.subheader("📄 Export")
            if st.button("Générer Rapport PDF"):
                try:
                    # Capture image sans crash
                    img_bytes = fig.to_image(format="png", engine="kaleido")
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(190, 10, f"Rapport Quantitatif : {name_display}", ln=True, align='C')
                    pdf.image(io.BytesIO(img_bytes), x=10, y=25, w=190)
                    
                    pdf.set_y(135)
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(190, 10, "Synthese des indicateurs :", ln=True)
                    pdf.set_font("Arial", '', 11)
                    pdf.cell(90, 8, f"Score de Qualite : {score_q:.1f}/10", border=1)
                    pdf.cell(100, 8, f"Position Sigma : {sig_pos:.2f}", border=1, ln=True)
                    pdf.cell(90, 8, f"CAGR Historique : {cagr:.2%}", border=1)
                    pdf.cell(100, 8, f"Fiabilite (R2) : {r2:.4f}", border=1, ln=True)
                    
                    st.download_button("⬇️ Télécharger PDF", data=pdf.output(dest='S'), file_name=f"{selected_ticker}_Analyse.pdf")
                except Exception as e:
                    st.error(f"Erreur PDF : {e}")

    else:
        st.error("Données insuffisantes pour ce ticker.")
