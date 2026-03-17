import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile
import os

st.set_page_config(page_title="Analyse Quantitative Pro", layout="wide")

# --- FONCTION DE CALCUL CORE ---
def run_analysis(ticker, name, reg_mode):
    data = yf.download(ticker, start="2000-01-01", progress=False)
    if data.empty or len(data) < 30: return None
    
    df = data[['Close']].copy().dropna().reset_index()
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    vol = df['Log_Ret'].std() * np.sqrt(252)
    
    df['Idx'] = np.arange(len(df)).reshape(-1, 1)
    X = df['Idx'].values.reshape(-1, 1)
    y = np.log(df['Close'].values) if reg_mode == "Logarithmique" else df['Close'].values
    
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = model.score(X, y)
    std_dev = np.std(y - y_pred)
    
    years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
    cagr = (df['Close'].iloc[-1] / df['Close'].iloc[0])**(1/years) - 1
    cur_y = np.log(df['Close'].iloc[-1]) if reg_mode == "Logarithmique" else df['Close'].iloc[-1]
    sig_pos = (cur_y - y_pred[-1]) / std_dev
    
    # Score Qualité
    s_r2 = r2 * 4
    s_cagr = min(max(cagr * 20, 0), 4)
    s_vol = max(2 - (vol * 2), 0)
    score = s_r2 + s_cagr + s_vol
    
    return {
        "df": df, "y_pred": y_pred, "std_dev": std_dev, "r2": r2, 
        "cagr": cagr, "vol": vol, "sig_pos": sig_pos, "score": score,
        "ticker": ticker, "name": name, "model": model
    }

# --- INTERFACE ---
st.sidebar.header("Configuration")
method = st.sidebar.radio("Source :", ("Saisie Manuelle", "Fichier Excel"))

tickers_to_process = [] # Liste pour le mode Batch

if method == "Fichier Excel":
    file = st.sidebar.file_uploader("Charger Excel", type="xlsx")
    if file:
        df_excel = pd.read_excel(file)
        cols = df_excel.columns.tolist()
        t_col = st.sidebar.selectbox("Colonne Tickers", cols)
        n_col = next((c for c in cols if "nom" in c.lower()), None)
        
        # Pour l'affichage unitaire
        ticker_list = df_excel[t_col].dropna().unique().tolist()
        selected_ticker = st.sidebar.selectbox("Action", ticker_list)
        name_display = df_excel[df_excel[t_col] == selected_ticker][n_col].iloc[0] if n_col else selected_ticker
        
        # Pour le mode Batch
        for _, row in df_excel.iterrows():
            tickers_to_process.append((row[t_col], row[n_col] if n_col else row[t_col]))
else:
    selected_ticker = st.sidebar.text_input("Ticker", "MSFT").upper()
    name_display = selected_ticker
    tickers_to_process = [(selected_ticker, selected_ticker)]

reg_mode = st.sidebar.radio("Modèle :", ("Logarithmique", "Linéaire"))

if selected_ticker:
    res = run_analysis(selected_ticker, name_display, reg_mode)
    
    if res:
        # Affichage Graphique (identique au précédent)
        def rev(v): return np.exp(v) if reg_mode == "Logarithmique" else v
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']+2*res['std_dev']), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']-2*res['std_dev']), fill='tonexty', fillcolor='rgba(255,0,0,0.05)', line_color='rgba(0,0,0,0)', name="±2σ"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']+res['std_dev']), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']-res['std_dev']), fill='tonexty', fillcolor='rgba(0,200,0,0.1)', line_color='rgba(0,0,0,0)', name="±1σ"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']), line=dict(color='orange', width=1, dash='dash'), name="Tendance"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=res['df']['Close'], line=dict(color='black', width=2), name="Prix"))
        
        fig.update_layout(template="plotly_white", paper_bgcolor='white', plot_bgcolor='white', yaxis_type="log" if reg_mode == "Logarithmique" else "linear", yaxis=dict(side="right"))
        st.plotly_chart(fig, use_container_width=True)

        # Affichage Métriques
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", f"{res['cagr']:.2%}")
        c2.metric("Volatilité", f"{res['vol']:.2%}")
        c3.metric("R²", f"{res['r2']:.3f}")
        c4.metric("SCORE QUALITÉ", f"{res['score']:.1f} / 10")

        # --- EXPORT PDF ---
        st.divider()
        col_pdf1, col_pdf2 = st.columns(2)
        
        with col_pdf1:
            if st.button("📄 Générer Rapport PDF (Action Seule)"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, f"Rapport Quantitatif : {res['name']} ({res['ticker']})", ln=True, align='C')
                pdf.set_font("Arial", '', 12)
                pdf.ln(10)
                pdf.cell(200, 10, f"Score Qualite : {res['score']:.1f} / 10", ln=True)
                pdf.cell(200, 10, f"CAGR : {res['cagr']:.2%} | Volatilite : {res['vol']:.2%}", ln=True)
                pdf.cell(200, 10, f"Position actuelle : {res['sig_pos']:.2f} sigma", ln=True)
                
                # Sauvegarde temporaire du graphique
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.write_image(tmpfile.name)
                    pdf.image(tmpfile.name, x=10, y=60, w=190)
                
                st.download_button("⬇️ Télécharger le PDF", data=pdf.output(dest='S').encode('latin-1'), file_name=f"Analyse_{res['ticker']}.pdf")

        with col_pdf2:
            if method == "Fichier Excel" and st.button("🗂️ Générer Rapport Batch (Tout l'Excel)"):
                pdf_b = FPDF()
                for t, n in tickers_to_process:
                    data_res = run_analysis(t, n, reg_mode)
                    if data_res:
                        pdf_b.add_page()
                        pdf_b.set_font("Arial", 'B', 16)
                        pdf_b.cell(200, 10, f"Analyse : {n} ({t})", ln=True)
                        pdf_b.set_font("Arial", '', 11)
                        pdf_b.cell(200, 8, f"Score: {data_res['score']:.1f}/10 | CAGR: {data_res['cagr']:.1%} | Sigma: {data_res['sig_pos']:.2f}", ln=True)
                        pdf_b.ln(5)
                st.download_button("⬇️ Télécharger Rapport Batch", data=pdf_b.output(dest='S').encode('latin-1'), file_name="Rapport_Batch_Complet.pdf")
