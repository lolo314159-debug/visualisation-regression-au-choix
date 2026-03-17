import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from fpdf import FPDF
import io

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="Analyse Quantitative Pro", layout="wide")

# --- FONCTION DE CALCUL BLINDÉE ---
def run_analysis(ticker, name, reg_mode):
    try:
        data = yf.download(ticker, start="2000-01-01", progress=False)
        if data.empty or len(data) < 30: return None
        
        # Nettoyage colonne Close
        df_price = data['Close'].copy().dropna()
        if isinstance(df_price, pd.DataFrame): df_price = df_price.iloc[:, 0]
        
        df = df_price.reset_index()
        df.columns = ['Date', 'Close']
        
        # Valeurs scalaires pures
        last_p = float(df['Close'].iloc[-1])
        first_p = float(df['Close'].iloc[0])
        
        # Volatilité (Log Returns)
        log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        vol = float(log_ret.std() * np.sqrt(252))
        
        # Régression Linéaire ou Log
        X = np.arange(len(df)).reshape(-1, 1)
        y = np.log(df['Close'].values) if reg_mode == "Logarithmique" else df['Close'].values
        
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X).flatten()
        r2 = float(model.score(X, y))
        std_dev = float(np.std(y - y_pred))
        
        # CAGR et Position Sigma
        years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
        cagr = float((last_p / first_p)**(1/years) - 1)
        
        cur_y_val = np.log(last_p) if reg_mode == "Logarithmique" else last_p
        sig_pos = float((cur_y_val - y_pred[-1]) / std_dev)
        
        # Score Qualité (R² 4pts, CAGR 4pts, Vol 2pts)
        s_r2 = r2 * 4
        s_cagr = np.clip(cagr * 20, 0, 4)
        s_vol = np.clip(2 - (vol * 2), 0, 2)
        score = float(s_r2 + s_cagr + s_vol)
        
        return {
            "df": df, "y_pred": y_pred, "std_dev": std_dev, "r2": r2, 
            "cagr": cagr, "vol": vol, "sig_pos": sig_pos, "score": score,
            "ticker": ticker, "name": str(name), "last_p": last_p
        }
    except Exception as e:
        st.error(f"Erreur sur {ticker}: {e}")
        return None

# --- BARRE LATÉRALE ---
st.sidebar.header("Configuration")
method = st.sidebar.radio("Source :", ("Saisie Manuelle", "Fichier Excel"))

tickers_to_process = []
selected_ticker = ""
name_display = ""

if method == "Fichier Excel":
    file = st.sidebar.file_uploader("Charger Excel", type="xlsx")
    if file:
        df_excel = pd.read_excel(file)
        cols = [str(c) for c in df_excel.columns]
        t_col = next((c for c in cols if "ticker" in c.lower()), cols[0])
        n_col = next((c for c in cols if "nom" in c.lower()), t_col)
        
        ticker_list = df_excel[t_col].dropna().astype(str).unique().tolist()
        selected_ticker = st.sidebar.selectbox("Action à analyser", ticker_list)
        name_display = str(df_excel[df_excel[t_col] == selected_ticker][n_col].iloc[0])
        for _, row in df_excel.iterrows():
            tickers_to_process.append((str(row[t_col]), str(row[n_col])))
else:
    selected_ticker = st.sidebar.text_input("Ticker (ex: MSFT, OR.PA)", "MSFT").upper()
    name_display = selected_ticker
    tickers_to_process = [(selected_ticker, selected_ticker)]

reg_mode = st.sidebar.radio("Modèle de croissance :", ("Logarithmique", "Linéaire"))

# --- ANALYSE ET AFFICHAGE ---
if selected_ticker:
    res = run_analysis(selected_ticker, name_display, reg_mode)
    
    if res:
        def rev(v): return np.exp(v) if reg_mode == "Logarithmique" else v
        
        # Graphique Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']+2*res['std_dev']), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']-2*res['std_dev']), fill='tonexty', fillcolor='rgba(255,0,0,0.05)', name="±2σ (Achat/Vente)"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']), line=dict(color='orange', width=1, dash='dash'), name="Tendance Long Terme"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=res['df']['Close'], line=dict(color='black', width=1.5), name="Prix de Clôture"))
        
        fig.update_layout(
            title=dict(text=f"Analyse : {res['name']}", font=dict(size=24)),
            template="plotly_white",
            yaxis=dict(side="right", type="log" if reg_mode == "Logarithmique" else "linear"),
            margin=dict(l=0, r=0, t=50, b=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Métriques principales
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CAGR (Croissance)", f"{res['cagr']:.2%}")
        m2.metric("Volatilité (Risque)", f"{res['vol']:.2%}")
        m3.metric("Fiabilité (R²)", f"{res['r2']:.3f}")
        m4.metric("SCORE QUALITÉ", f"{res['score']:.1f} / 10")

        # Diagnostic
        st.divider()
        st.subheader("📝 Diagnostic Quantitatif")
        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            st.write(f"**Positionnement :** {res['sig_pos']:.2f} σ")
            if abs(res['sig_pos']) > 2:
                st.error("⚠️ Anomalie Statistique : Le prix est hors de son corridor historique.")
            elif abs(res['sig_pos']) > 1:
                st.warning("⚖️ Déviation : Le prix s'écarte de sa moyenne.")
            else:
                st.success("✅ Zone Neutre : Le prix suit sa tendance théorique.")

        # --- EXPORT PDF AVEC GRAPHIQUE ---
        with c_diag2:
            if st.button("📄 Générer Rapport PDF Complet"):
                try:
                    # Capture de l'image (nécessite kaleido)
                    img_bytes = fig.to_image(format="png", width=800, height=450)
                    
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", 'B', 18)
                    pdf.cell(190, 15, f"Rapport Quantitatif : {res['name']}", ln=True, align='C')
                    
                    # Insertion Image
                    img_stream = io.BytesIO(img_bytes)
                    pdf.image(img_stream, x=10, y=30, w=190)
                    
                    # Tableau de données
                    pdf.set_y(140)
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(190, 10, "Indicateurs de Performance :", ln=True)
                    pdf.set_font("Arial", '', 11)
                    data_table = [
                        ["Indicateur", "Valeur", "Interprétation"],
                        ["Score de Qualité", f"{res['score']:.1f}/10", "Qualité intrinsèque"],
                        ["Position Sigma", f"{res['sig_pos']:.2f}", "Écart à la moyenne"],
                        ["CAGR", f"{res['cagr']:.2%}", "Rendement annuel"],
                        ["Volatilité", f"{res['vol']:.2%}", "Niveau de risque"],
                        ["R² (Fiabilité)", f"{res['r2']:.3f}", "Régularité"]
                    ]
                    for row in data_table:
                        pdf.cell(50, 8, row[0], border=1)
                        pdf.cell(40, 8, row[1], border=1)
                        pdf.cell(100, 8, row[2], border=1, ln=True)

                    st.download_button("⬇️ Télécharger le PDF", data=pdf.output(dest='S').encode('latin-1'), file_name=f"Analyse_{res['ticker']}.pdf")
                except Exception as e:
                    st.error(f"Erreur d'export PDF : {e}. Vérifiez l'installation de Kaleido.")

        # --- MODE BATCH (Excel seulement) ---
        if method == "Fichier Excel":
            st.divider()
            if st.button("🗂️ Analyser toute la liste (Batch)"):
                batch_list = []
                bar = st.progress(0)
                for i, (t, n) in enumerate(tickers_to_process):
                    a = run_analysis(t, n, reg_mode)
                    if a: batch_list.append(a)
                    bar.progress((i + 1) / len(tickers_to_process))
                
                # Tri par opportunité (Sigma le plus bas en premier)
                batch_list.sort(key=lambda x: x['sig_pos'])
                df_res = pd.DataFrame([{
                    "Ticker": x['ticker'], "Nom": x['name'], 
                    "Sigma": round(x['sig_pos'], 2), "Score": round(x['score'], 1),
                    "Prix": round(x['last_p'], 2)
                } for x in batch_list])
                st.table(df_res)
