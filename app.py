import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from fpdf import FPDF
import io

st.set_page_config(page_title="Analyse Quantitative", layout="wide")

# --- CONFIGURATION & SOURCE ---
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

# --- LOGIQUE DE CALCUL (Blindage contre ValueError et Scalaires) ---
if selected_ticker:
    ticker_obj = yf.Ticker(selected_ticker)
    data = ticker_obj.history(start="2000-01-01")
    
    if not name_display:
        try: name_display = ticker_obj.info.get('longName', selected_ticker)
        except: name_display = selected_ticker

    if not data.empty and len(data) > 30:
        df = data[['Close']].copy().dropna().reset_index()
        
        # Correction ValueError (image 2) : Extraction propre des valeurs
        X = np.arange(len(df)).reshape(-1, 1)
        y = np.log(df['Close'].values) if reg_mode == "Logarithmique" else df['Close'].values
        
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X).flatten()
        r2 = float(model.score(X, y))
        std_dev = float(np.std(y - y_pred))
        
        def rev(v): return np.exp(v) if reg_mode == "Logarithmique" else v

        # --- GRAPHIQUE ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred + 2*std_dev), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred - 2*std_dev), fill='tonexty', fillcolor='rgba(255, 0, 0, 0.05)', line_color='rgba(0,0,0,0)', name="Zone ±2σ"))
        fig.add_trace(go.Scatter(x=df['Date'], y=rev(y_pred), line=dict(color='orange', width=1, dash='dash'), name="Tendance"))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], line=dict(color='black', width=2), name="Prix"))

        fig.update_layout(
            title=f"<b>{name_display}</b> | {selected_ticker}",
            template="plotly_white", paper_bgcolor='white', plot_bgcolor='white',
            yaxis=dict(side="right"), xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- CALCULS ANALYTIQUES ---
        last_p = float(df['Close'].iloc[-1])
        years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
        cagr = float((last_p / df['Close'].iloc[0])**(1/years) - 1)
        
        # Position Sigma (image 3 fix)
        cur_y = np.log(last_p) if reg_mode == "Logarithmique" else last_p
        sig_pos = float((cur_y - y_pred[-1]) / std_dev)

        # --- INTERFACE OBJECTIFS (Structure app (1).py) ---
        st.subheader("🎯 Objectifs et Supports Théoriques")
        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("Vente (+2σ)", f"{rev(y_pred[-1] + 2*std_dev):.2f}")
        o2.metric("Objectif (+1σ)", f"{rev(y_pred[-1] + std_dev):.2f}")
        o3.metric("Moyenne", f"{rev(y_pred[-1]):.2f}")
        o4.metric("Support (-1σ)", f"{rev(y_pred[-1] - std_dev):.2f}")
        o5.metric("Achat (-2σ)", f"{rev(y_pred[-1] - 2*std_dev):.2f}")

        # --- ANALYSE ET EXPORT ---
        st.divider()
        col_diag, col_pdf = st.columns([2, 1])
        
        with col_diag:
            st.subheader("📝 Diagnostic de l'Investisseur")
            # Calcul du Score de Qualité (Nouveauté)
            score_q = (r2 * 5) + (np.clip(cagr * 20, 0, 3)) + (np.clip(2 - (std_dev*10), 0, 2))
            
            st.write(f"**Score de Qualité :** {score_q:.1f} / 10")
            st.write(f"**Fiabilité du modèle (R²) :** {r2:.4f}")
            
            if sig_pos < -1.5:
                st.success(f"🔥 **ACHAT STATISTIQUE :** Décote majeure ({sig_pos:.2f}σ).")
            elif sig_pos > 1.5:
                st.error(f"🚨 **SURCHAUFFE :** Extension dangereuse (+{sig_pos:.2f}σ).")
            else:
                st.info(f"⚖️ **NEUTRE :** Évolution normative ({sig_pos:.2f}σ).")

        with col_pdf:
            st.subheader("📤 Export")
            if st.button("Générer Rapport PDF"):
                try:
                    # Correction image (image 4 fix) : bypass du fichier temporaire
                    img_bytes = fig.to_image(format="png", engine="kaleido")
                    
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(190, 10, f"Rapport Quantitatif : {name_display}", ln=True, align='C')
                    pdf.image(io.BytesIO(img_bytes), x=10, y=30, w=190)
                    
                    pdf.set_y(140)
                    pdf.set_font("Arial", '', 11)
                    pdf.cell(190, 10, f"Dernier Prix : {last_p:.2f} | CAGR : {cagr:.2%}", ln=True)
                    pdf.cell(190, 10, f"Score Qualite : {score_q:.1f}/10 | Position : {sig_pos:.2f} sigma", ln=True)
                    
                    # Correction format binaire (image 6 fix)
                    pdf_data = pdf.output(dest='S')
                    if isinstance(pdf_data, bytearray):
                        pdf_data = bytes(pdf_data)
                        
                    st.download_button("⬇️ Télécharger", data=pdf_data, file_name=f"{selected_ticker}_Analyse.pdf")
                except Exception as e:
                    st.error(f"Erreur PDF : {e}")
    else:
        st.error("Données insuffisantes.")
