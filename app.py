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
        
        # Nettoyage pour n'avoir que des chiffres (Série 1D)
        df_price = data['Close'].copy().dropna()
        if isinstance(df_price, pd.DataFrame): 
            df_price = df_price.iloc[:, 0]
        
        df = df_price.reset_index()
        df.columns = ['Date', 'Close']
        
        # Forcer le passage en float pour éviter les erreurs de scalaires Pandas
        last_p = float(df['Close'].iloc[-1])
        first_p = float(df['Close'].iloc[0])
        
        # Volatilité
        log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        vol = float(log_ret.std() * np.sqrt(252))
        
        # Régression
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
        sig_pos = float((cur_y_val - float(y_pred[-1])) / std_dev)
        
        # Score Qualité avec np.clip pour la stabilité
        score = float((r2 * 4) + np.clip(cagr * 20, 0, 4) + np.clip(2 - (vol * 2), 0, 2))
        
        return {
            "df": df, "y_pred": y_pred, "std_dev": std_dev, "r2": r2, 
            "cagr": cagr, "vol": vol, "sig_pos": sig_pos, "score": score,
            "ticker": ticker, "name": str(name), "last_p": last_p
        }
    except Exception as e:
        st.sidebar.error(f"Erreur technique : {e}")
        return None

# --- INTERFACE ---
st.sidebar.header("Configuration")
method = st.sidebar.radio("Source :", ("Saisie Manuelle", "Fichier Excel"))

tickers_to_process = []
if method == "Fichier Excel":
    file = st.sidebar.file_uploader("Charger Excel", type="xlsx")
    if file:
        df_excel = pd.read_excel(file)
        cols = [str(c) for c in df_excel.columns]
        t_col = next((c for c in cols if "ticker" in c.lower()), cols[0])
        n_col = next((c for c in cols if "nom" in c.lower()), t_col)
        ticker_list = df_excel[t_col].dropna().astype(str).unique().tolist()
        selected_ticker = st.sidebar.selectbox("Action", ticker_list)
        name_display = str(df_excel[df_excel[t_col] == selected_ticker][n_col].iloc[0])
        for _, row in df_excel.iterrows():
            tickers_to_process.append((str(row[t_col]), str(row[n_col])))
else:
    selected_ticker = st.sidebar.text_input("Ticker", "MSFT").upper()
    name_display = selected_ticker
    tickers_to_process = [(selected_ticker, selected_ticker)]

reg_mode = st.sidebar.radio("Modèle :", ("Logarithmique", "Linéaire"))

if selected_ticker:
    res = run_analysis(selected_ticker, name_display, reg_mode)
    if res:
        def rev(v): return np.exp(v) if reg_mode == "Logarithmique" else v
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']+2*res['std_dev']), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']-2*res['std_dev']), fill='tonexty', fillcolor='rgba(255,0,0,0.05)', name="±2σ"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']), line=dict(color='orange', width=1, dash='dash'), name="Tendance"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=res['df']['Close'], line=dict(color='black', width=1.5), name="Prix"))
        fig.update_layout(template="plotly_white", yaxis=dict(side="right", type="log" if reg_mode == "Logarithmique" else "linear"))
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CAGR", f"{res['cagr']:.2%}")
        m2.metric("Volatilité", f"{res['vol']:.2%}")
        m3.metric("R²", f"{res['r2']:.3f}")
        m4.metric("SCORE", f"{res['score']:.1f}/10")

        # --- EXPORT PDF CORRIGÉ ---
        if st.button("📄 Télécharger Rapport PDF"):
            try:
                # Capture d'image sécurisée
                img_bytes = fig.to_image(format="png", engine="kaleido")
                
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", 'B', 16)
                pdf.cell(190, 10, f"Analyse : {res['name']}", ln=True, align='C')
                
                # Ajout de l'image via un stream
                img_stream = io.BytesIO(img_bytes)
                pdf.image(img_stream, x=10, y=30, w=190)
                
                pdf.set_y(140)
                pdf.set_font("Helvetica", '', 12)
                pdf.cell(190, 10, f"Score : {res['score']:.1f}/10 | Position : {res['sig_pos']:.2f} sigma", ln=True)
                pdf.cell(190, 10, f"CAGR : {res['cagr']:.2%} | Volatilite : {res['vol']:.2%}", ln=True)
                
                # Sortie PDF en bytes directement (évite l'erreur .encode())
                pdf_output = pdf.output() 
                
                st.download_button(
                    label="Confirmer le téléchargement",
                    data=pdf_output,
                    file_name=f"{res['ticker']}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Erreur d'export : {e}")
