import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from fpdf import FPDF
import io

# ... (garder la fonction run_analysis identique à la précédente) ...

if selected_ticker:
    res = run_analysis(selected_ticker, name_display, reg_mode)
    if res:
        def rev(v): return np.exp(v) if reg_mode == "Logarithmique" else v
        
        # --- CRÉATION DU GRAPHIQUE ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']+2*res['std_dev']), line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']-2*res['std_dev']), fill='tonexty', fillcolor='rgba(255,0,0,0.05)', name="±2σ"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=rev(res['y_pred']), line=dict(color='orange', width=1, dash='dash'), name="Tendance"))
        fig.add_trace(go.Scatter(x=res['df']['Date'], y=res['df']['Close'], line=dict(color='black', width=1.5), name="Prix"))
        
        fig.update_layout(
            title=f"Régression : {res['name']}",
            template="plotly_white",
            yaxis=dict(side="right"),
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- GÉNÉRATION DU PDF AVEC IMAGE ---
        if st.button("📄 Télécharger Rapport PDF Complet"):
            try:
                # 1. Convertir le graphique Plotly en image (PNG) en mémoire
                # On utilise engine="kaleido" explicitement
                img_bytes = fig.to_image(format="png", width=800, height=450, engine="kaleido")
                
                # 2. Création du PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 18)
                pdf.cell(190, 15, f"Rapport Quantitatif : {res['name']}", ln=True, align='C')
                
                # Insertion de l'image (BytesIO permet de ne pas créer de fichier physique)
                img_stream = io.BytesIO(img_bytes)
                pdf.image(img_stream, x=10, y=30, w=190)
                
                # Données sous le graphique
                pdf.set_y(140) 
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(190, 10, "Indicateurs Clés :", ln=True)
                pdf.set_font("Arial", '', 11)
                pdf.cell(95, 8, f"Score de Qualité : {res['score']:.1f} / 10", border=1)
                pdf.cell(95, 8, f"Position : {res['sig_pos']:.2f} sigma", border=1, ln=True)
                pdf.cell(95, 8, f"CAGR : {res['cagr']:.2%}", border=1)
                pdf.cell(95, 8, f"Volatilité : {res['vol']:.2%}", border=1, ln=True)
                pdf.cell(95, 8, f"Fiabilité (R2) : {res['r2']:.3f}", border=1)
                pdf.cell(95, 8, f"Dernier Prix : {res['last_p']:.2f}", border=1, ln=True)

                # 3. Export
                pdf_output = pdf.output(dest='S')
                st.download_button(
                    label="⬇️ Cliquez ici pour finaliser le téléchargement",
                    data=bytes(pdf_output),
                    file_name=f"Analyse_{res['ticker']}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"Erreur lors de la génération de l'image : {e}")
                st.info("Note : Si Kaleido échoue sur le serveur, vérifiez que 'kaleido' est bien dans requirements.txt")
