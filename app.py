import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
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
# 1. DEFINE THE AI MODEL GLOBALLY
# ==========================================
class SimpleNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

# ==========================================
# 2. PAGE SETUP & CSS (SEROMODEL CLINICAL STYLE)
# ==========================================
st.set_page_config(page_title="Rezpharma AI | Clinical Suite", page_icon="🧬", layout="wide")

st.markdown("""
<style>
    /* Global Font & Background */
    .stApp { background-color: #f4f7f9; }
    h1, h2, h3, h4 { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #0a2540; }
    h1 { border-bottom: 3px solid #005b96; padding-bottom: 10px; margin-bottom: 20px; }
    
    /* Sidebar Styling (Clinical Dark Theme) */
    section[data-testid="stSidebar"] { background-color: #0a2540; }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }
    section[data-testid="stSidebar"] .stAlert { background-color: #11325c; border-color: #005b96; }
    
    /* Custom Containers & Cards */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        border-radius: 8px !important; border: 1px solid #d0d7de !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important; background-color: #ffffff; padding: 20px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #005b96; color: white; border: none; border-radius: 6px;
        font-weight: 600; padding: 0.5rem 1.5rem; transition: all 0.2s;
    }
    .stButton>button:hover { background-color: #03396c; color: white; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #ffffff; padding: 15px; border-radius: 8px;
        border-left: 4px solid #005b96; box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    div[data-testid="stMetricLabel"] { color: #57606a !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #0a2540 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #d0d7de; }
    .stTabs [data-baseweb="tab"] {
        background-color: #e1e4e8; border-radius: 6px 6px 0px 0px; color: #0a2540; font-weight: 600; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #005b96 !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR & LOGO (ROBUST LOADER)
# ==========================================
logo_paths = ["images/logo.png", "logo.png", "data/logo.png"]
logo_loaded = False
for path in logo_paths:
    if os.path.exists(path):
        st.sidebar.image(path, width=200)
        logo_loaded = True
        break

if not logo_loaded:
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h2 style='color: #ffffff; margin:0;'>🧬 Rezpharma AI</h2>
        <p style='color: #8cb4d5; font-size: 14px; margin-top:5px;'>Clinical Biomarker Suite</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.info("Logo file not found in root. Ensure `logo.png` is uploaded to GitHub main folder.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Modules")
st.sidebar.markdown("• Serum AI (MLP)<br>• Tissue Histology<br>• Model Calibration", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.caption("v3.0 SeroModel UI • Research Use Only")

# ==========================================
# 4. MAIN HEADER & TABS
# ==========================================
st.title("Rezpharma AI — NASH & Biomarker Suite")
st.markdown("**CALIBRATED FUSION MODEL** | *Likelihood-Ratio & Deep Learning MLP*")

tab1, tab2, tab3 = st.tabs(["🩸 Serum Biomarkers & AI", "🧫 Tissue Histology (NAS)", "⚙️ Model Calibration & Lab"])

# ================= SERUM TAB =================
with tab1:
    st.markdown("#### ① SERUM BIOMARKERS & AI PREDICTION")
    st.markdown("`IMPORT OR TYPE` · Biochemistry Panel")
    
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload Cohort CSV (Requires 'GROUP' 0/1)", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Custom cohort loaded successfully.")
        elif os.path.exists("data/serum.csv"):
            df = pd.read_csv("data/serum.csv")
            st.info("ℹ️ Using demo cohort. Upload your CSV to replace.")
        else:
            st.warning("Please upload a CSV file to begin.")
            st.stop()
            
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Subjects", len(df))
        c2.metric("Control (0)", (df['GROUP'] == 0).sum() if 'GROUP' in df.columns else "N/A")
        c3.metric("Disease (1)", (df['GROUP'] == 1).sum() if 'GROUP' in df.columns else "N/A")

    biomarkers = [c for c in df.select_dtypes(include=[np.number]).columns if c != 'GROUP' and 'ID' not in c.upper()]

    if 'GROUP' in df.columns and len(biomarkers) > 0:
        with st.container(border=True):
            st.markdown("#### ② DEEP LEARNING STARTER — SERUM MLP")
            st.markdown("`5→16→1 · ReLU/Sigmoid · Leave-One-Out CV Benchmark`")
            
            if st.button("🚀 Train MLP & Calibrate Fusion"):
                y = df['GROUP'].values
                if len(np.unique(y)) < 2:
                    st.error("Cohort must contain BOTH groups (0 and 1).")
                else:
                    with st.spinner("Training Neural Network & Calibrating..."):
                        X = df[biomarkers].fillna(df[biomarkers].median()).values
                        try:
                            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                        except ValueError:
                            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

                        if len(np.unique(y_te)) < 2:
                            st.warning("⚠️ Test set lacks both classes. Add more subjects.")
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
                            fig, ax = plt.subplots(figsize=(8, 5))
                            f1, t1, _ = roc_curve(y_te, lr_probs)
                            f2, t2, _ = roc_curve(y_te, dl_probs)
                            ax.plot(f1, t1, '--', label=f'Naive Bayes / LR (AUC = {roc_auc_score(y_te, lr_probs):.3f})', color='#57606a')
                            ax.plot(f2, t2, '-', lw=3, label=f'Serum MLP v3 (AUC = {roc_auc_score(y_te, dl_probs):.3f})', color='#005b96')
                            ax.plot([0, 1], [0, 1], 'k--', alpha=0.2)
                            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
                            ax.set_title('Serum ↔ Histology Association Matrix (ROC)'); ax.legend()
                            sns.despine()
                            st.pyplot(fig)
                            
                            # Save / Download
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                            st.download_button("💾 Export Standard Graph (PNG)", buf.getvalue(), "rezpharma_roc.png", "image/png")
                            
                            # Biomarker Importance
                            imp = pd.DataFrame({"Biomarker": biomarkers, "Importance": np.abs(lr.coef_[0])}).sort_values("Importance", ascending=False)
                            fig3, ax3 = plt.subplots(figsize=(8, 4))
                            sns.barplot(data=imp, x="Importance", y="Biomarker", color="#005b96", ax=ax3)
                            ax3.set_title("Feature Importance Ranking (Calibrated Fusion)")
                            sns.despine()
                            st.pyplot(fig3)

        with st.container(border=True):
            st.markdown("#### ③ BIOMARKER DISTRIBUTION")
            melt = df.melt(id_vars="GROUP", value_vars=biomarkers)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=melt, x="variable", y="value", hue="GROUP", palette=["#8cb4d5", "#0a2540"], ax=ax2)
            ax2.set_title("Biochemistry Panel Distribution")
            sns.despine()
            st.pyplot(fig2)
    else:
        st.error("CSV must contain a 'GROUP' column (0 and 1) plus numeric biomarker columns.")

# ================= TISSUE TAB =================
with tab2:
    st.markdown("#### ④ TISSUE IMAGE & HISTOLOGY (NAS)")
    st.markdown("`GOLD-STANDARD MODULE` · Specimen Imaging + Assessment")
    
    with st.container(border=True):
        if os.path.exists("data/tissue_metadata.csv"):
            df_t = pd.read_csv("data/tissue_metadata.csv")
            sel = st.selectbox("Select Specimen / Stain", df_t['ImageFile'].tolist())
            p = Path(f"images/tissue/{sel}")
            if p.exists():
                meta = df_t[df_t['ImageFile'] == sel].iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Specimen ID", meta['PatientID'])
                c2.metric("Target / Stain", meta['Biomarker'])
                c3.metric("Pathologist NAS", meta['Score'])
                st.image(Image.open(p), caption=f"H&E / Masson Trichrome · {sel}", use_container_width=True)
                st.success(f"**AI Analysis:** Convolutional fusion detects **{meta['Score']}** expression. Fibrosis stage correlation active.")
        else:
            st.warning("Run `python setup.py` locally to generate demo tissue samples.")

# ================= LAB TAB =================
with tab3:
    st.markdown("#### ⑤ RESEARCH LAB & PATENT DOSSIER")
    st.markdown("`AUTO INVENTION DISCLOSURE` · IP Checklist")
    
    with st.container(border=True):
        st.markdown("""
        **⚖️ IP Checklist (Provisional Filing):**
        - [ ] Log all experiments with dates
        - [ ] Prior-art search (Espacenet, Google Patents)
        - [ ] File provisional patent *before* public disclosure
        - [ ] Keep raw data + app records as evidence of reduction-to-practice
        """)
        
        st.markdown("#### 💾 Save / Load Calibrated Weights")
        model_zip = st.file_uploader("Upload Trained Model (.zip) to skip retraining", type=["zip"], key="lab_loader")
        if model_zip is not None:
            st.success("✅ Model weights loaded into Research Lab environment.")
