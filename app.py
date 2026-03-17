import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile

st.set_page_config(page_title="Analyse Quantitative Pro", layout="wide")

# --- FONCTION DE CALCUL (Correction des types de données) ---
def run_analysis(ticker, name, reg_mode):
    try:
        data = yf.download(ticker, start="2000-01-01", progress=False)
        if data.empty or len(data) < 30: return None
        
        df = data[['Close']].copy().dropna().reset_index()
        
        # Extraction de valeurs scalaires pures pour éviter les erreurs de Series
        last_p = float(df['Close'].iloc[-1])
        first_p = float(df['Close'].iloc[0])
        
        # Calcul Volatilité
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        vol = float(df['Log_Ret'].std() * np.sqrt(252))
        
        # Régression
        df['Idx'] = np.arange(len(df)).reshape(-1, 1)
        X = df['Idx'].values.reshape(-1, 1)
        y = np.log(df['Close'].values) if reg_mode == "Logarithmique" else df['Close'].values
        
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = float(model.score(X, y))
        std_dev = float(np.std(y - y_pred))
        
        # CAGR et Sigma
        days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
        years = days / 365.25
        cagr = float((last_p / first_p)**(1/years) - 1)
        
        cur_y_val = np.log(last_p) if reg_mode == "Logarithmique" else last_p
        sig_pos = float((cur_y_val - float(y_pred[-1])) / std_dev)
        
        # Score Qualité (Correction de la ligne qui plantait)
        s_r2 = r2 * 4
        # Utilisation de max/min natif Python sur des floats
        s_cagr = float(max(0, min(cagr * 20, 4))) 
        s_vol = float(max(0, min(2 - (vol * 2), 2)))
        score = float(s_r2 + s_cagr + s_vol)
        
        return {
            "df": df, "y_pred": y_pred, "std_dev": std_dev, "r2": r2, 
            "cagr": cagr, "vol": vol, "sig_pos": sig_pos, "score": score,
            "ticker": ticker, "name": str(name), "last_p": last_p
        }
    except Exception as e:
        st.sidebar.error(f"Erreur sur {ticker}: {e}")
        return None

# --- INTERFACE ---
st.sidebar.header("Configuration")
method = st.sidebar.radio("Source :", ("Saisie Manuelle", "Fichier Excel"))

tickers_to_process = []

if method == "Fichier Excel":
    file = st.sidebar.file_uploader("Charger Excel", type="xlsx")
    if file:
        df_excel = pd.read_excel(file)
        cols = df_excel.columns.tolist()
        # On cherche dynamiquement les colonnes pour éviter les KeyError
        t_col = next((c for c in cols if "ticker" in c.lower()), cols[0])
        n_col = next((c for c in cols if "nom" in c.lower()), t_col)
        
        ticker_list = df_excel[t_col].dropna().astype(str).unique().tolist()
        selected_ticker = st.sidebar.selectbox("Action à afficher", ticker_list)
        name_display = df_excel[df_excel[t_col] == selected_ticker][n_col].iloc[0]
        
        for _, row in df_excel.iterrows():
            tickers_to_process.append((str(row[t_col]), str(row[n_col])))
else:
    selected_ticker = st.sidebar.text_input("Ticker", "MSFT").upper()
    name_display = selected_ticker
    tickers_to_process = [(selected_ticker, selected_ticker)]

reg_mode = st.sidebar.radio("Modèle :", ("Logarithmique", "Linéaire"))

# --- AFFICHAGE ---
if selected_ticker:
    res = run_analysis(selected_ticker, name_display, reg_mode)
    
    if res:
        def rev(v): return np.exp(v) if reg_mode == "Logarithmique" else v
        
        # Graphique
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']+2*res['std_dev']), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']-2*res['std_dev']), fill='tonexty', fillcolor='rgba(255,0,0,0.05)', line_color='rgba(0,0,0,0)', name="±2σ"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']+res['std_dev']), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']-res['std_dev']), fill='tonexty', fillcolor='rgba(0,200,0,0.1)', line_color='rgba(0,0,0,0)', name="±1σ"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']), line=dict(color='orange', width=1, dash='dash'), name="Tendance"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=res['df']['Close'], line=dict(color='black', width=2), name="Prix"))
        
        fig.update_layout(title=f"<b>{res['name']}</b> ({res['ticker']})", template="plotly_white", yaxis_type="log" if reg_mode == "Logarithmique" else "linear", yaxis=dict(side="right"))
        st.plotly_chart(fig, use_container_width=True)

        # Métriques
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CAGR", f"{res['cagr']:.2%}")
        m2.metric("Volatilité", f"{res['vol']:.2%}")
        m3.metric("R²", f"{res['r2']:.3f}")
        m4.metric("SCORE QUALITÉ", f"{res['score']:.1f} / 10")

        # Bouton PDF
        if st.button("📄 Télécharger Rapport PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, f"Analyse : {res['name']}", ln=True, align='C')
            pdf.set_font("Arial", '', 12)
            pdf.ln(10)
            pdf.cell(200, 10, f"Score : {res['score']:.1f}/10 | Position : {res['sig_pos']:.2f} sigma", ln=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                fig.write_image(tmp.name)
                pdf.image(tmp.name, x=10, y=50, w=190)
            st.download_button("Confirmer le téléchargement", data=pdf.output(dest='S').encode('latin-1'), file_name=f"{res['ticker']}.pdf")
    else:
        st.error("Impossible d'analyser ce titre. Vérifiez le ticker ou la connexion.")
