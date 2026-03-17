import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from fpdf import FPDF
import tempfile

st.set_page_config(page_title="Analyse Quantitative Pro", layout="wide")

# --- FONCTION DE CALCUL CORE (CORRIGÉE) ---
def run_analysis(ticker, name, reg_mode):
    try:
        data = yf.download(ticker, start="2000-01-01", progress=False)
        if data.empty or len(data) < 30: return None
        
        df = data[['Close']].copy().dropna().reset_index()
        # Conversion explicite pour éviter les erreurs de type Series/Scalar
        last_price = float(df['Close'].iloc[-1])
        first_price = float(df['Close'].iloc[0])
        
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        vol = float(df['Log_Ret'].std() * np.sqrt(252))
        
        df['Idx'] = np.arange(len(df)).reshape(-1, 1)
        X = df['Idx'].values.reshape(-1, 1)
        y = np.log(df['Close'].values) if reg_mode == "Logarithmique" else df['Close'].values
        
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        r2 = float(model.score(X, y))
        std_dev = float(np.std(y - y_pred))
        
        years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
        cagr = float((last_price / first_price)**(1/years) - 1)
        cur_y = np.log(last_price) if reg_mode == "Logarithmique" else last_price
        sig_pos = float((cur_y - y_pred[-1]) / std_dev)
        
        # Score Qualité (Logic corrigée)
        s_r2 = r2 * 4
        s_cagr = min(max(cagr * 20, 0), 4)
        s_vol = max(2 - (vol * 2), 0)
        score = float(s_r2 + s_cagr + s_vol)
        
        return {
            "df": df, "y_pred": y_pred, "std_dev": std_dev, "r2": r2, 
            "cagr": cagr, "vol": vol, "sig_pos": sig_pos, "score": score,
            "ticker": ticker, "name": name, "last_price": last_price
        }
    except:
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
        # Correction KeyError: cherche 'Ticker' ou la première colonne
        t_col = 'Ticker' if 'Ticker' in cols else cols[0]
        n_col = next((c for c in cols if "nom" in c.lower()), t_col)
        
        ticker_list = df_excel[t_col].dropna().unique().tolist()
        selected_ticker = st.sidebar.selectbox("Action", ticker_list)
        name_display = df_excel[df_excel[t_col] == selected_ticker][n_col].iloc[0]
        
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
        # Graphique
        def rev(v): return np.exp(v) if reg_mode == "Logarithmique" else v
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']+2*res['std_dev']), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']-2*res['std_dev']), fill='tonexty', fillcolor='rgba(255,0,0,0.05)', line_color='rgba(0,0,0,0)', name="±2σ"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']+res['std_dev']), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']-res['std_dev']), fill='tonexty', fillcolor='rgba(0,200,0,0.1)', line_color='rgba(0,0,0,0)', name="±1σ"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']), line=dict(color='orange', width=1, dash='dash'), name="Tendance"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=res['df']['Close'], line=dict(color='black', width=2), name="Prix"))
        fig.update_layout(title=f"<b>{res['name']}</b>", template="plotly_white", yaxis_type="log" if reg_mode == "Logarithmique" else "linear", yaxis=dict(side="right"))
        st.plotly_chart(fig, use_container_width=True)

        # Métriques
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", f"{res['cagr']:.2%}")
        c2.metric("Volatilité", f"{res['vol']:.2%}")
        c3.metric("R²", f"{res['r2']:.3f}")
        c4.metric("SCORE QUALITÉ", f"{res['score']:.1f} / 10")

        # --- EXPORTS ---
        st.divider()
        col_pdf1, col_pdf2 = st.columns(2)
        with col_pdf1:
            if st.button("📄 Rapport PDF Action"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16); pdf.cell(200, 10, f"Analyse : {res['name']}", ln=True, align='C')
                pdf.set_font("Arial", '', 12); pdf.ln(10)
                pdf.cell(200, 10, f"Score Qualite : {res['score']:.1f}/10 | Position : {res['sig_pos']:.2f} sigma", ln=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    fig.write_image(tmp.name); pdf.image(tmp.name, x=10, y=50, w=190)
                st.download_button("⬇️ Télécharger PDF", data=pdf.output(dest='S').encode('latin-1'), file_name=f"{res['ticker']}.pdf")

        with col_pdf2:
            if method == "Fichier Excel" and st.button("🗂️ Rapport Batch Trié"):
                batch_results = []
                with st.spinner("Calcul des opportunités..."):
                    for t, n in tickers_to_process:
                        analysis = run_analysis(t, n, reg_mode)
                        if analysis: batch_results.append(analysis)
                
                # Tri par Sigma croissant (les plus sous-évaluées en premier)
                batch_results.sort(key=lambda x: x['sig_pos'])
                
                pdf_b = FPDF()
                pdf_b.add_page()
                pdf_b.set_font("Arial", 'B', 14); pdf_b.cell(200, 10, "Classement par Opportunite Statistique (Sigma)", ln=True)
                pdf_b.set_font("Courier", '', 9)
                pdf_b.ln(5)
                pdf_b.cell(200, 5, f"{'TICKER':<10} {'NOM':<25} {'SIGMA':<10} {'SCORE':<8} {'PRIX':<10}", ln=True)
                pdf_b.cell(200, 2, "-"*65, ln=True)
                
                for r in batch_results:
                    line = f"{r['ticker']:<10} {r['name'][:24]:<25} {r['sig_pos']:<10.2f} {r['score']:<8.1f} {r['last_price']:<10.2f}"
                    pdf_b.cell(200, 5, line, ln=True)
                
                st.download_button("⬇️ Télécharger Liste Triée", data=pdf_b.output(dest='S').encode('latin-1'), file_name="Opportunites_Batch.pdf")
