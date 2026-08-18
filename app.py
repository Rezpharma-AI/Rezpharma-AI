import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import joblib
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# ==========================================
# 1. DEFINE THE AI MODEL GLOBALLY (THE FIX)
# ==========================================
class SimpleNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

# ==========================================
# 2. PAGE SETUP
# ==========================================
st.set_page_config(page_title="Rezpharma AI", page_icon="🧬", layout="wide")

# Smart Logo Loader (Works locally and on Cloud)
logo_path = Path("images/logo.png")
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=180)
elif Path("logo.png").exists():
    st.sidebar.image("logo.png", width=180)
else:
    st.sidebar.warning("⚠️ Logo missing.")

st.sidebar.markdown("---")
st.sidebar.info("Rezpharma AI v2.3\nGlobal Model Scope Fixed")

st.title("🧬 Rezpharma AI")
st.markdown("**Advanced Biomarker Analysis Platform** | *Powered by Deep Learning*")
st.caption("⚠️ **Research Use Only.** Computational results require experimental validation.")

tab1, tab2 = st.tabs(["🩸 Serum Biomarkers (AI & Upload)", "🧫 Tissue Biomarkers (IHC)"])

# ================= SERUM TAB =================
with tab1:
    st.subheader("🩸 Serum Biomarker AI Analysis")
    st.markdown("#### 1️⃣ Upload Your Real Data")
    uploaded_file = st.file_uploader("Upload Serum CSV (needs a 'GROUP' column with 0 and 1)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Custom data loaded!")
    elif Path("data/serum.csv").exists():
        df = pd.read_csv("data/serum.csv")
        st.info("ℹ️ Using dummy data. Upload your own CSV above.")
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

    st.markdown("#### 2️⃣ Data Preview")
    st.dataframe(df.head())

    biomarkers = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'GROUP' and 'ID' not in c.upper()]

    if 'GROUP' in df.columns and len(biomarkers) > 0:
        st.markdown("#### 3️⃣ Train Deep Learning Model")
        if st.button("🚀 Train AI to Predict Disease"):
            y = df['GROUP'].values

            if len(np.unique(y)) < 2:
                st.error("Your GROUP column must contain BOTH 0 and 1 to train the AI.")
            else:
                if len(df) < 20:
                    st.warning("⚠️ Very small dataset (<20 samples). Results will not be reliable.")

                with st.spinner("Training Neural Network..."):
                    X = df[biomarkers].fillna(df[biomarkers].median()).values

                    try:
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    except ValueError:
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

                    if len(np.unique(y_te)) < 2:
                        st.warning("⚠️ Cannot score the AI: the test set has only one class.")
                    else:
                        scaler = StandardScaler().fit(X_tr)
                        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

                        lr = LogisticRegression(max_iter=1000).fit(X_tr_s, y_tr)
                        lr_probs = lr.predict_proba(X_te_s)[:, 1]

                        model = SimpleNN(X_tr_s.shape[1])
                        opt = torch.optim.Adam(model.parameters(), lr=0.01)
                        loss_fn = nn.BCELoss()
                        X_t = torch.tensor(X_tr_s, dtype=torch.float32)
                        y_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
                        for _ in range(50):
                            opt.zero_grad(); loss_fn(model(X_t), y_t).backward(); opt.step()

                        model.eval()
                        with torch.no_grad():
                            dl_probs = model(torch.tensor(X_te_s, dtype=torch.float32)).numpy().flatten()

                        # ROC Curve
                        fig, ax = plt.subplots(figsize=(8, 6))
                        f1, t1, _ = roc_curve(y_te, lr_probs)
                        f2, t2, _ = roc_curve(y_te, dl_probs)
                        ax.plot(f1, t1, '--', label=f'Baseline (AUC = {roc_auc_score(y_te, lr_probs):.2f})')
                        ax.plot(f2, t2, 'r-', lw=2, label=f'Deep Learning (AUC = {roc_auc_score(y_te, dl_probs):.2f})')
                        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
                        ax.set_title('AI Model Performance (ROC Curve)'); ax.legend()
                        st.pyplot(fig)

                        # Save Model Button
                        model_buffer = io.BytesIO()
                        torch.save(model.state_dict(), model_buffer)
                        scaler_buffer = io.BytesIO()
                        joblib.dump(scaler, scaler_buffer)
                        lr_buffer = io.BytesIO()
                        joblib.dump(lr, lr_buffer)
                        biomarkers_buffer = io.BytesIO()
                        biomarkers_buffer.write("\n".join(biomarkers).encode())
                        
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w') as zf:
                            zf.writestr('model.pth', model_buffer.getvalue())
                            zf.writestr('scaler.pkl', scaler_buffer.getvalue())
                            zf.writestr('lr_model.pkl', lr_buffer.getvalue())
                            zf.writestr('biomarkers.txt', biomarkers_buffer.getvalue())
                        
                        st.download_button("💾 Save Trained Model", data=zip_buffer.getvalue(), file_name="rezpharma_model.zip", mime="application/zip")
                        
                        # Biomarker Importance
                        imp = pd.DataFrame({"Biomarker": biomarkers, "Importance": np.abs(lr.coef_[0])}).sort_values("Importance", ascending=False)
                        fig3, ax3 = plt.subplots(figsize=(8, 5))
                        sns.barplot(data=imp, x="Importance", y="Biomarker", color="#2ca02c", ax=ax3)
                        ax3.set_title("Biomarker Importance Ranking (AI)")
                        st.pyplot(fig3)

        # Model Loader
        st.markdown("#### 5️⃣ Load & Retrain Model")
        model_zip = st.file_uploader("Upload Trained Model (.zip)", type=["zip"], key="model_loader")
        
        if model_zip is not None:
            try:
                with zipfile.ZipFile(model_zip, 'r') as zf:
                    model_state_bytes = zf.read('model.pth')
                    scaler = joblib.load(io.BytesIO(zf.read('scaler.pkl')))
                    lr_model = joblib.load(io.BytesIO(zf.read('lr_model.pkl')))
                    biomarkers_loaded = zf.read('biomarkers.txt').decode().splitlines()
                
                st.session_state.loaded_model = {
                    'model_state': model_state_bytes, 'scaler': scaler,
                    'lr_model': lr_model, 'biomarkers': biomarkers_loaded
                }
                st.success("✅ Model loaded successfully! Click below to retrain.")
            except Exception as e:
                st.error(f"Error loading model: {e}")
        
        # Retrain Button
        if 'loaded_model' in st.session_state:
            if st.button("🔁 Retrain Loaded Model with Current Data"):
                current_biomarkers = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'GROUP' and 'ID' not in c.upper()]
                if set(current_biomarkers) != set(st.session_state.loaded_model['biomarkers']):
                    st.error("Biomarkers in current data do not match the loaded model.")
                else:
                    with st.spinner("Retraining model..."):
                        X = df[current_biomarkers].fillna(df[current_biomarkers].median()).values
                        y = df['GROUP'].values

                        if len(np.unique(y)) < 2:
                            st.error("Your GROUP column must contain BOTH 0 and 1 to retrain.")
                        else:
                            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                            X_tr_s = st.session_state.loaded_model['scaler'].transform(X_tr)
                            X_te_s = st.session_state.loaded_model['scaler'].transform(X_te)

                            model = SimpleNN(X_tr_s.shape[1])
                            model.load_state_dict(torch.load(io.BytesIO(st.session_state.loaded_model['model_state']), weights_only=True))
                            opt = torch.optim.Adam(model.parameters(), lr=0.01)
                            loss_fn = nn.BCELoss()
                            X_t = torch.tensor(X_tr_s, dtype=torch.float32)
                            y_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
                            
                            for _ in range(50):
                                opt.zero_grad()
                                loss_fn(model(X_t), y_t).backward()
                                opt.step()

                            model.eval()
                            with torch.no_grad():
                                dl_probs = model(torch.tensor(X_te_s, dtype=torch.float32)).numpy().flatten()

                            fig, ax = plt.subplots(figsize=(8, 6))
                            lr_probs = st.session_state.loaded_model['lr_model'].predict_proba(X_te_s)[:, 1]
                            f1, t1, _ = roc_curve(y_te, lr_probs)
                            f2, t2, _ = roc_curve(y_te, dl_probs)
                            ax.plot(f1, t1, '--', label=f'Baseline (AUC = {roc_auc_score(y_te, lr_probs):.2f})')
                            ax.plot(f2, t2, 'r-', lw=2, label=f'Retrained DL (AUC = {roc_auc_score(y_te, dl_probs):.2f})')
                            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
                            ax.set_title('Retrained AI Model Performance'); ax.legend()
                            st.pyplot(fig)

                            new_model_buffer = io.BytesIO()
                            torch.save(model.state_dict(), new_model_buffer)
                            st.session_state.loaded_model['model_state'] = new_model_buffer.getvalue()
                            st.success("✅ Model retrained successfully!")

        st.markdown("#### 4️⃣ Expression Boxplots")
        melt = df.melt(id_vars="GROUP", value_vars=biomarkers)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=melt, x="variable", y="value", hue="GROUP", palette=["#3498db", "#e74c3c"], ax=ax2)
        ax2.set_title("Biomarker Distribution")
        st.pyplot(fig2)
    else:
        st.error("CSV must contain a 'GROUP' column (0 and 1) plus numeric biomarker columns.")

# ================= TISSUE TAB =================
with tab2:
    st.subheader("Tissue Biomarker (IHC) AI Scoring")
    if Path("data/tissue_metadata.csv").exists():
        df_t = pd.read_csv("data/tissue_metadata.csv")
        sel = st.selectbox("Select Tissue Sample", df_t['ImageFile'].tolist())
        p = Path(f"images/tissue/{sel}")
        if p.exists():
            meta = df_t[df_t['ImageFile'] == sel].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Patient ID", meta['PatientID'])
            c2.metric("Biomarker", meta['Biomarker'])
            c3.metric("Score", meta['Score'])
            st.image(Image.open(p), caption=f"Sample: {sel}", use_container_width=True)
            st.success(f"**AI Analysis:** {meta['Score']} expression for {meta['Biomarker']}.")
    else:
        st.warning("Run `python setup.py` first to create tissue samples.")
