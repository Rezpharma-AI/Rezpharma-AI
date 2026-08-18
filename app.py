import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import curve_fit
import io
import base64
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, recall_score, confusion_matrix
import pickle

# ==========================================
# PAGE CONFIG & GLOBAL STYLES
# ==========================================
st.set_page_config(
    page_title="RezPharma AI Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look and code‑generated logo
st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3 { color: #0a2540; font-family: 'Segoe UI', Roboto, sans-serif; }
    .stButton>button {
        background-color: #005b96; color: white; border-radius: 6px;
        font-weight: 600; padding: 0.5rem 1.5rem; border: none;
    }
    .stButton>button:hover { background-color: #03396c; }
    .formal-box {
        background: linear-gradient(135deg, #0a2540 0%, #1a3a5c 100%);
        color: white; padding: 30px; border-radius: 12px;
        margin-bottom: 30px; text-align: center;
    }
    .formal-box h1 { color: white; font-size: 2.5rem; margin-bottom: 10px; }
    .formal-box p { color: #d0d7de; font-size: 1.1rem; }
    div[data-testid="stMetric"] {
        background: white; padding: 15px; border-radius: 8px;
        border-left: 4px solid #005b96; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #e1e4e8; border-radius: 6px 6px 0 0;
        color: #0a2540; font-weight: 600; padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] { background: #005b96; color: white; }
    .sidebar-logo {
        text-align: center;
        padding: 20px 0;
    }
    .sidebar-logo .logo-icon {
        font-size: 48px;
        display: block;
    }
    .sidebar-logo .logo-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 5px;
    }
    .sidebar-logo .logo-subtitle {
        color: #8cb4d5;
        font-size: 0.9rem;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CODE‑GENERATED LOGO IN SIDEBAR
# ==========================================
st.sidebar.markdown("""
<div class="sidebar-logo">
    <span class="logo-icon">🧬</span>
    <div class="logo-title">RezPharma AI</div>
    <div class="logo-subtitle">Open Source Pharmaceutical Suite</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🧪 Modules
- In Vivo
- In Vitro
- Clinical
- Stats & Graph Maker
- Deep Learning & Multimodal AI
- Deep Biomarker Analyzer
- Sign Up
""")
st.sidebar.markdown("---")
st.sidebar.caption("v1.0 • Research Use Only")

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def download_link(df, filename="data.csv", text="Download CSV"):
    """Generate a download link for a DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def hill_equation(x, bottom, top, ic50, hill_slope):
    """Four-parameter logistic (Hill) equation for dose-response."""
    return bottom + (top - bottom) / (1 + (x / ic50) ** hill_slope)

# ==========================================
# PAGE FUNCTIONS
# ==========================================

def home():
    st.markdown("""
    <div class="formal-box">
        <h1>🔬 RezPharma AI Platform</h1>
        <p><strong>AI for Scientific Studies in Pharmaceutical Science & Medicine</strong></p>
        <p style="font-size:1rem;">
            This platform provides advanced computational tools for <em>in vivo</em>, <em>in vitro</em>, and <em>clinical</em> research.
            It is intended for <strong>research and educational purposes only</strong>.
            All results must be validated by qualified professionals before clinical or regulatory use.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🧪 In Vivo Modules", "3+")
        st.markdown("Animal models, PK/PD, toxicology")
    with col2:
        st.metric("🧫 In Vitro Modules", "3+")
        st.markdown("Dose‑response, assay processing, viability")
    with col3:
        st.metric("🩺 Clinical Modules", "4+")
        st.markdown("Cohort analysis, biomarker discovery, risk models")

    st.markdown("---")
    st.markdown("""
    ## 🚀 Getting Started
    1. Use the **sidebar navigation** to switch between modules.
    2. Upload your own data (CSV, Excel, or images) where required.
    3. Explore the tools and download publication‑ready graphs.

    ## 📚 Available Modules
    - **In Vivo**: Statistical analysis of animal data, non‑compartmental PK, survival curves.
    - **In Vitro**: IC50/EC50 fitting, plate reader normalization, Z‑factor.
    - **Clinical**: Patient cohort summaries, ML classification, biomarker feature importance.
    - **Stats & Graph Maker**: Prism‑style interactive plots, t‑tests, ANOVA, correlation.
    - **Deep Learning & Multimodal AI**: Image classification, text mining, multimodal fusion.
    - **Deep Biomarker Analyzer**: Per‑biomarker statistics, correlations, and neural network latent space.
    - **Sign Up**: Create a free account to save your work (coming soon).
    """)

def in_vivo():
    st.title("🧪 In Vivo Studies")
    st.markdown("Tools for animal model data, pharmacokinetics, and toxicology.")

    tab1, tab2, tab3 = st.tabs(["📊 Animal Model Analyzer", "💊 PK/PD (NCA)", "☠️ Toxicology Dashboard"])

    with tab1:
        st.subheader("Animal Model Data Analysis")
        uploaded = st.file_uploader("Upload CSV with columns: Group, BodyWeight, OrganWeights, Biomarkers", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success("Data loaded.")
            st.dataframe(df.head())

            group_col = st.selectbox("Grouping column", df.columns)
            value_col = st.selectbox("Value column", df.select_dtypes(include=np.number).columns)

            if group_col and value_col:
                groups = df[group_col].unique()
                if len(groups) >= 2:
                    if len(groups) == 2:
                        g1 = df[df[group_col] == groups[0]][value_col].dropna()
                        g2 = df[df[group_col] == groups[1]][value_col].dropna()
                        t_stat, p_val = stats.ttest_ind(g1, g2)
                        st.markdown(f"**T‑test p‑value:** {p_val:.4f}")
                    else:
                        groups_data = [df[df[group_col] == g][value_col].dropna() for g in groups]
                        f_stat, p_val = stats.f_oneway(*groups_data)
                        st.markdown(f"**ANOVA p‑value:** {p_val:.4f}")

                    fig = px.box(df, x=group_col, y=value_col, points="all")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Please upload a CSV file to begin.")

    with tab2:
        st.subheader("Non‑Compartmental PK Analysis (NCA)")
        st.markdown("Upload time‑concentration data (Time, Concentration).")
        uploaded = st.file_uploader("Upload PK data CSV", type="csv", key="pk")
        if uploaded:
            pk_df = pd.read_csv(uploaded)
            st.dataframe(pk_df.head())
            if {'Time', 'Concentration'}.issubset(pk_df.columns):
                t = pk_df['Time'].values
                c = pk_df['Concentration'].values
                auc = np.trapz(c, t)
                cmax = c.max()
                tmax = t[np.argmax(c)]
                st.metric("AUC (0‑t)", f"{auc:.2f}")
                st.metric("Cmax", f"{cmax:.2f}")
                st.metric("Tmax", f"{tmax:.2f}")
            else:
                st.warning("CSV must contain 'Time' and 'Concentration' columns.")
        else:
            st.info("Upload PK data to compute AUC, Cmax, Tmax.")

    with tab3:
        st.subheader("Toxicology Dashboard")
        st.markdown("Upload survival data (Time, Event, Group) or dose‑toxicity data (Dose, Toxicity).")

        tox_option = st.radio("Select analysis type", ["Survival (Kaplan‑Meier)", "Dose‑Toxicity Modelling"])

        if tox_option == "Survival (Kaplan‑Meier)":
            surv_file = st.file_uploader("Upload survival CSV (Time, Event, Group)", type="csv", key="surv")
            if surv_file:
                surv_df = pd.read_csv(surv_file)
                st.dataframe(surv_df.head())

                required_cols = {'Time', 'Event', 'Group'}
                if required_cols.issubset(surv_df.columns):
                    from lifelines import KaplanMeierFitter
                    from lifelines.statistics import logrank_test
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(8, 5))
                    kmf = KaplanMeierFitter()
                    for group in surv_df['Group'].unique():
                        group_data = surv_df[surv_df['Group'] == group]
                        kmf.fit(durations=group_data['Time'], event_observed=group_data['Event'], label=str(group))
                        kmf.plot_survival_function(ax=ax)

                    ax.set_title("Kaplan‑Meier Survival Curves")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Survival Probability")
                    st.pyplot(fig)

                    groups = surv_df['Group'].unique()
                    if len(groups) == 2:
                        g1 = surv_df[surv_df['Group'] == groups[0]]
                        g2 = surv_df[surv_df['Group'] == groups[1]]
                        results = logrank_test(g1['Time'], g2['Time'], g1['Event'], g2['Event'])
                        st.metric("Log‑rank p‑value", f"{results.p_value:.4f}")
                else:
                    st.warning("CSV must contain columns: Time, Event, Group")
            else:
                st.info("Upload survival data to generate Kaplan‑Meier curves.")

        else:  # Dose‑Toxicity Modelling
            tox_file = st.file_uploader("Upload dose‑toxicity CSV (Dose, Toxicity)", type="csv", key="tox")
            if tox_file:
                tox_df = pd.read_csv(tox_file)
                st.dataframe(tox_df.head())

                if {'Dose', 'Toxicity'}.issubset(tox_df.columns):
                    x = tox_df['Dose'].values
                    y = tox_df['Toxicity'].values

                    try:
                        p0 = [np.min(y), np.max(y), np.median(x), 1]
                        popt, _ = curve_fit(hill_equation, x, y, p0=p0, maxfev=5000)
                        bottom, top, ld50, hill = popt
                        st.success(f"Estimated LD50: **{ld50:.3f}**")
                        st.write(f"Hill slope: {hill:.3f}")

                        x_smooth = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 100)
                        y_smooth = hill_equation(x_smooth, *popt)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data'))
                        fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='Fit'))
                        fig.update_layout(xaxis_title="Dose", yaxis_title="Toxicity")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Fitting failed: {e}")
                else:
                    st.warning("CSV must contain 'Dose' and 'Toxicity' columns.")
            else:
                st.info("Upload dose‑toxicity data to fit an LD50 model.")

def in_vitro():
    st.title("🧫 In Vitro Studies")
    st.markdown("Tools for cell‑based assays, dose‑response, and plate reader data.")

    tab1, tab2, tab3 = st.tabs(["📈 Dose‑Response Curve Fitter", "🔬 Assay Data Processor", "🧮 Z‑Factor Calculator"])

    with tab1:
        st.subheader("IC50 / EC50 Determination")
        st.markdown("Upload CSV with columns: Concentration, Response (or Inhibition %).")
        uploaded = st.file_uploader("Upload dose‑response data", type="csv")
        if uploaded:
            dr_df = pd.read_csv(uploaded)
            st.dataframe(dr_df.head())
            if {'Concentration', 'Response'}.issubset(dr_df.columns):
                x = dr_df['Concentration'].values
                y = dr_df['Response'].values

                try:
                    p0 = [np.min(y), np.max(y), np.median(x), 1]
                    popt, _ = curve_fit(hill_equation, x, y, p0=p0, maxfev=5000)
                    bottom, top, ic50, hill = popt
                    st.success(f"Fitted IC50: **{ic50:.3f}** (concentration units)")
                    st.write(f"Hill slope: {hill:.3f}")

                    x_smooth = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 100)
                    y_smooth = hill_equation(x_smooth, *popt)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data'))
                    fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='Fit'))
                    fig.update_layout(xaxis_title="Concentration", yaxis_title="Response")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Fitting failed: {e}")
            else:
                st.warning("CSV must contain 'Concentration' and 'Response' columns.")
        else:
            st.info("Upload dose‑response data to fit a 4‑parameter logistic curve.")

    with tab2:
        st.subheader("Plate Reader Data Processor")
        st.markdown("Normalise raw plate reader data and calculate Z‑factor from uploaded CSV.")
        assay_file = st.file_uploader("Upload raw assay CSV", type="csv", key="assay")
        if assay_file:
            assay_df = pd.read_csv(assay_file)
            st.dataframe(assay_df.head())

            st.markdown("**Select columns:**")
            pos_col = st.selectbox("Positive Control column", assay_df.columns)
            neg_col = st.selectbox("Negative Control column", assay_df.columns)
            sample_cols = st.multiselect("Sample columns (for normalisation)", assay_df.columns)

            if st.button("Process Data"):
                pos_mean = assay_df[pos_col].mean()
                neg_mean = assay_df[neg_col].mean()

                pos_sd = assay_df[pos_col].std()
                neg_sd = assay_df[neg_col].std()
                z_factor = 1 - (3*(pos_sd + neg_sd)) / abs(pos_mean - neg_mean)
                st.metric("Z‑factor", f"{z_factor:.3f}")

                if sample_cols:
                    norm_df = assay_df[sample_cols].apply(
                        lambda x: (x - neg_mean) / (pos_mean - neg_mean) * 100
                    )
                    st.subheader("Normalised Data (% of Positive Control)")
                    st.dataframe(norm_df)

                    fig = px.box(norm_df, points="all", title="Normalised Assay Values")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown(download_link(norm_df, "normalised_assay.csv", "Download Normalised CSV"), unsafe_allow_html=True)
                else:
                    st.warning("Please select at least one sample column to normalise.")
        else:
            st.info("Upload raw assay data (e.g., plate reader output) to process.")

    with tab3:
        st.subheader("Z‑Factor Calculator")
        st.markdown("Enter mean and SD of positive and negative controls to compute Z‑factor.")
        col1, col2 = st.columns(2)
        with col1:
            mean_pos = st.number_input("Mean Positive Control", value=100.0)
            sd_pos = st.number_input("SD Positive Control", value=10.0)
        with col2:
            mean_neg = st.number_input("Mean Negative Control", value=10.0)
            sd_neg = st.number_input("SD Negative Control", value=5.0)
        if st.button("Calculate Z‑factor"):
            z = 1 - (3*(sd_pos + sd_neg)) / abs(mean_pos - mean_neg)
            st.metric("Z‑factor", f"{z:.3f}")
            if z > 0.5:
                st.success("Excellent assay quality.")
            elif z > 0:
                st.warning("Marginal assay quality.")
            else:
                st.error("Poor assay – Z‑factor ≤ 0.")

def clinical():
    st.title("🩺 Clinical Studies")
    st.markdown("Patient cohort analysis, biomarker discovery, and risk prediction using machine learning.")

    st.subheader("Biomarker Discovery and Classification")
    uploaded = st.file_uploader("Upload clinical CSV (must contain 'GROUP' 0/1)", type="csv", key="clinical_csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Data loaded.")
        st.dataframe(df.head())

        if 'GROUP' not in df.columns:
            st.error("The CSV must contain a 'GROUP' column with 0/1 values.")
            return

        if {'AST', 'ALT'}.issubset(df.columns):
            df['AST_ALT_Ratio'] = df['AST'] / df['ALT'].replace(0, np.nan)
        if {'TG', 'HDL-C'}.issubset(df.columns):
            df['TG_HDL_Ratio'] = df['TG'] / df['HDL-C'].replace(0, np.nan)
        if {'AST', 'PLT'}.issubset(df.columns):
            df['APRI'] = ((df['AST'] / 40) / df['PLT'].replace(0, np.nan)) * 100

        biomarkers = [c for c in df.select_dtypes(include=np.number).columns if c != 'GROUP']

        st.subheader("Correlation Matrix")
        corr = df[biomarkers].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Train AI Model (Logistic Regression + PyTorch MLP)")
        if st.button("🚀 Train Model"):
            X = df[biomarkers].fillna(df[biomarkers].median()).values
            y = df['GROUP'].values

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            if len(np.unique(y_te)) < 2:
                st.error("Test set contains only one class. Cannot compute ROC.")
                return

            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_te_s = scaler.transform(X_te)

            lr = LogisticRegression(max_iter=1000).fit(X_tr_s, y_tr)
            lr_probs = lr.predict_proba(X_te_s)[:, 1]
            lr_auc = roc_auc_score(y_te, lr_probs)

            class MLP(nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(64, 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    )
                def forward(self, x):
                    return self.net(x)

            X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
            X_te_t = torch.tensor(X_te_s, dtype=torch.float32)

            model = MLP(X_tr_s.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            loss_fn = nn.BCELoss()

            best_loss = np.inf
            patience = 10
            epochs_no_improve = 0
            epochs = 200

            progress_bar = st.progress(0)
            status_text = st.empty()

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                train_loss = loss_fn(model(X_tr_t), y_tr_t)
                train_loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(model(X_tr_t), y_tr_t)

                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss.item():.4f}")

                if val_loss.item() < best_loss - 1e-4:
                    best_loss = val_loss.item()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        status_text.text(f"Early stopping at epoch {epoch+1}")
                        break

            model.eval()
            with torch.no_grad():
                dl_probs = model(X_te_t).numpy().flatten()
            dl_auc = roc_auc_score(y_te, dl_probs)

            fpr_lr, tpr_lr, _ = roc_curve(y_te, lr_probs)
            fpr_dl, tpr_dl, _ = roc_curve(y_te, dl_probs)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode='lines',
                                         name=f'Logistic Regression (AUC={lr_auc:.3f})',
                                         line=dict(dash='dash', color='gray')))
            fig_roc.add_trace(go.Scatter(x=fpr_dl, y=tpr_dl, mode='lines',
                                         name=f'PyTorch MLP (AUC={dl_auc:.3f})',
                                         line=dict(color='#005b96', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                         line=dict(dash='dot', color='black'), showlegend=False))
            fig_roc.update_layout(title="ROC Curves", xaxis_title="False Positive Rate",
                                  yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)

            lr_pred = (lr_probs >= 0.5).astype(int)
            dl_pred = (dl_probs >= 0.5).astype(int)

            st.subheader("Model Performance Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Logistic Regression**")
                st.write(f"Accuracy: {accuracy_score(y_te, lr_pred):.3f}")
                st.write(f"Sensitivity: {recall_score(y_te, lr_pred):.3f}")
                st.write(f"Specificity: {recall_score(y_te, lr_pred, pos_label=0):.3f}")
            with col2:
                st.markdown("**PyTorch MLP**")
                st.write(f"Accuracy: {accuracy_score(y_te, dl_pred):.3f}")
                st.write(f"Sensitivity: {recall_score(y_te, dl_pred):.3f}")
                st.write(f"Specificity: {recall_score(y_te, dl_pred, pos_label=0):.3f}")

            fig_cm = go.Figure()
            fig_cm.add_trace(go.Heatmap(z=confusion_matrix(y_te, lr_pred), text=confusion_matrix(y_te, lr_pred), texttemplate="%{text}", colorscale='Blues'))
            fig_cm.update_layout(title="Confusion Matrix (Logistic Regression)")
            st.plotly_chart(fig_cm, use_container_width=True)

            fig_cm2 = go.Figure()
            fig_cm2.add_trace(go.Heatmap(z=confusion_matrix(y_te, dl_pred), text=confusion_matrix(y_te, dl_pred), texttemplate="%{text}", colorscale='Blues'))
            fig_cm2.update_layout(title="Confusion Matrix (MLP)")
            st.plotly_chart(fig_cm2, use_container_width=True)

            importance = pd.DataFrame({
                'Biomarker': biomarkers,
                'Coefficient': np.abs(lr.coef_[0])
            }).sort_values('Coefficient', ascending=True)
            fig_imp = px.bar(importance, x='Coefficient', y='Biomarker', orientation='h',
                             title="Feature Importance (Logistic Regression)")
            st.plotly_chart(fig_imp, use_container_width=True)

            st.session_state['clinical_artifacts'] = {
                'model': model,
                'scaler': scaler,
                'feature_names': biomarkers,
                'lr_model': lr
            }
            st.success("Model trained successfully! You can now use the Single Patient Prediction below.")

            # Download model
            artifacts_dict = {
                'scaler': scaler,
                'logistic_model': lr,
                'pytorch_model_state': model.state_dict(),
                'feature_names': biomarkers
            }
            buf = io.BytesIO()
            pickle.dump(artifacts_dict, buf)
            st.download_button("Download Model Pipeline (.pkl)", buf.getvalue(), "clinical_model.pkl", "application/octet-stream")

        if 'clinical_artifacts' in st.session_state:
            st.subheader("Single Patient Prediction")
            artifacts = st.session_state['clinical_artifacts']

            with st.form("single_patient_form"):
                input_data = {}
                cols = st.columns(3)
                for i, feat in enumerate(artifacts['feature_names']):
                    with cols[i % 3]:
                        default_val = float(df[feat].median()) if feat in df else 0.0
                        input_data[feat] = st.number_input(feat, value=default_val, step=0.01)
                predict_btn = st.form_submit_button("Predict Risk")

            if predict_btn:
                input_df = pd.DataFrame([input_data])
                input_scaled = artifacts['scaler'].transform(input_df)
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
                artifacts['model'].eval()
                with torch.no_grad():
                    prob = artifacts['model'](input_tensor).item()
                st.metric("Predicted Probability", f"{prob:.3f}")
                st.progress(min(int(prob * 100), 100))
    else:
        st.info("Upload clinical data to perform biomarker analysis and train AI models.")

def stats_graphs():
    st.title("📊 Statistical Tools & Graph Maker")
    st.markdown("Prism‑style interactive graphs and SPSS‑like statistical tests.")

    uploaded = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded:
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.success("Data loaded.")
        st.dataframe(df.head())

        st.subheader("Graph Builder")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X axis", df.columns)
        with col2:
            y_axis = st.selectbox("Y axis", df.columns)
        with col3:
            chart_type = st.selectbox("Chart type", ["Scatter", "Line", "Bar", "Box", "Violin", "Histogram"])

        if chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis)
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis)
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis)
        elif chart_type == "Box":
            fig = px.box(df, x=x_axis, y=y_axis)
        elif chart_type == "Violin":
            fig = px.violin(df, x=x_axis, y=y_axis)
        else:
            fig = px.histogram(df, x=x_axis)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Statistical Tests")
        test_type = st.selectbox("Select test", [
            "T‑test (two groups)", "Mann‑Whitney U (two groups)",
            "ANOVA (multiple groups)", "Kruskal‑Wallis (multiple groups)",
            "Pearson correlation", "Spearman correlation",
            "Chi‑square test"
        ])

        if test_type == "T‑test (two groups)":
            group_col = st.selectbox("Group column", df.columns)
            value_col = st.selectbox("Value column", df.select_dtypes(include=np.number).columns)
            groups = df[group_col].unique()
            if len(groups) == 2:
                g1 = df[df[group_col] == groups[0]][value_col].dropna()
                g2 = df[df[group_col] == groups[1]][value_col].dropna()
                t_stat, p_val = stats.ttest_ind(g1, g2)
                st.metric("p‑value", f"{p_val:.4f}")
                st.write(f"t‑statistic: {t_stat:.4f}")
            else:
                st.warning("T‑test requires exactly 2 groups.")

        elif test_type == "Mann‑Whitney U (two groups)":
            group_col = st.selectbox("Group column", df.columns)
            value_col = st.selectbox("Value column", df.select_dtypes(include=np.number).columns)
            groups = df[group_col].unique()
            if len(groups) == 2:
                g1 = df[df[group_col] == groups[0]][value_col].dropna()
                g2 = df[df[group_col] == groups[1]][value_col].dropna()
                u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                st.metric("p‑value", f"{p_val:.4f}")
                st.write(f"U‑statistic: {u_stat:.4f}")
            else:
                st.warning("Mann‑Whitney U requires exactly 2 groups.")

        elif test_type == "ANOVA (multiple groups)":
            group_col = st.selectbox("Group column", df.columns)
            value_col = st.selectbox("Value column", df.select_dtypes(include=np.number).columns)
            groups = df[group_col].unique()
            if len(groups) >= 2:
                groups_data = [df[df[group_col] == g][value_col].dropna() for g in groups]
                f_stat, p_val = stats.f_oneway(*groups_data)
                st.metric("p‑value", f"{p_val:.4f}")
                st.write(f"F‑statistic: {f_stat:.4f}")
            else:
                st.warning("ANOVA requires at least 2 groups.")

        elif test_type == "Kruskal‑Wallis (multiple groups)":
            group_col = st.selectbox("Group column", df.columns)
            value_col = st.selectbox("Value column", df.select_dtypes(include=np.number).columns)
            groups = df[group_col].unique()
            if len(groups) >= 2:
                groups_data = [df[df[group_col] == g][value_col].dropna() for g in groups]
                h_stat, p_val = stats.kruskal(*groups_data)
                st.metric("p‑value", f"{p_val:.4f}")
                st.write(f"H‑statistic: {h_stat:.4f}")
            else:
                st.warning("Kruskal‑Wallis requires at least 2 groups.")

        elif test_type == "Pearson correlation":
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) >= 2:
                col1 = st.selectbox("First variable", num_cols)
                col2 = st.selectbox("Second variable", num_cols)
                r, p = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                st.metric("Pearson r", f"{r:.4f}")
                st.metric("p‑value", f"{p:.4f}")
            else:
                st.warning("Need at least 2 numeric columns.")

        elif test_type == "Spearman correlation":
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) >= 2:
                col1 = st.selectbox("First variable", num_cols)
                col2 = st.selectbox("Second variable", num_cols)
                rho, p = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
                st.metric("Spearman rho", f"{rho:.4f}")
                st.metric("p‑value", f"{p:.4f}")
            else:
                st.warning("Need at least 2 numeric columns.")

        elif test_type == "Chi‑square test":
            col1 = st.selectbox("First categorical", df.columns)
            col2 = st.selectbox("Second categorical", df.columns)
            contingency = pd.crosstab(df[col1], df[col2])
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            st.metric("p‑value", f"{p:.4f}")
            st.write(f"Chi‑square statistic: {chi2:.4f}, dof={dof}")
            st.dataframe(contingency)

    else:
        st.info("Upload a dataset to begin creating graphs and running tests.")

def deep_learning():
    st.title("🧠 Deep Learning & Multimodal AI")
    st.markdown("Advanced AI tools for image analysis, text mining, and multimodal fusion.")

    tab1, tab2, tab3 = st.tabs(["🖼️ Image Classification", "📄 Text Mining", "🔗 Multimodal Fusion"])

    with tab1:
        st.subheader("Image Classification with Pre‑trained CNN (ResNet18)")
        st.markdown("Upload an image (e.g., histology, MRI) to classify it using a ResNet18 model pre‑trained on ImageNet.")

        uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="dl_image")
        if uploaded is not None:
            from PIL import Image
            import torchvision.transforms as transforms
            from torchvision import models

            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            try:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                model.eval()

                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                input_tensor = preprocess(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)

                import requests
                labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                try:
                    labels = requests.get(labels_url).text.splitlines()
                except:
                    labels = [f"Class {i}" for i in range(1000)]

                top5_prob, top5_catid = torch.topk(probabilities, 5)
                st.markdown("### Top Predictions:")
                for i in range(5):
                    st.write(f"{labels[top5_catid[i]]}: {top5_prob[i].item()*100:.2f}%")

            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.info("Make sure 'torch' and 'torchvision' are installed (check requirements.txt).")
        else:
            st.info("Upload an image to classify it.")

    with tab2:
        st.subheader("Text Mining & Document Analysis")
        st.markdown("Paste text or upload multiple .txt files to extract top keywords and analyze document similarity.")

        input_method = st.radio("Input method", ["Paste text", "Upload files"])

        documents = []
        if input_method == "Paste text":
            text_input = st.text_area("Enter text (one document per line or a single paragraph)", height=200)
            if text_input:
                docs = [doc.strip() for doc in text_input.split('\n') if doc.strip()]
                if not docs:
                    docs = [text_input.strip()]
                documents = docs
        else:
            uploaded_files = st.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)
            if uploaded_files:
                for file in uploaded_files:
                    documents.append(file.read().decode("utf-8"))

        if documents:
            st.success(f"Loaded {len(documents)} document(s).")
            st.subheader("Top Keywords (TF‑IDF)")

            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np

            vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()

            global_scores = tfidf_matrix.sum(axis=0).A1
            top_indices = global_scores.argsort()[::-1][:10]
            top_keywords = [(feature_names[i], global_scores[i]) for i in top_indices]

            if top_keywords:
                kw_df = pd.DataFrame(top_keywords, columns=["Keyword", "Score"])
                fig = px.bar(kw_df, x="Score", y="Keyword", orientation='h')
                st.plotly_chart(fig, use_container_width=True)

            if len(documents) > 1:
                st.subheader("Document Similarity Matrix")
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(tfidf_matrix)
                fig = px.imshow(sim_matrix, text_auto=True, aspect="auto",
                                labels=dict(x="Document", y="Document", color="Similarity"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Provide text or upload files to analyze.")

    with tab3:
        st.subheader("Multimodal Fusion – Tabular + Image (Demo)")
        st.markdown("Upload a CSV (one row per sample) and optionally a ZIP of images (one per sample).")
        csv_file = st.file_uploader("Upload tabular CSV (with 'target' column)", type="csv", key="mm_csv2")
        zip_file = st.file_uploader("Upload ZIP of images (optional)", type="zip", key="mm_zip")

        if csv_file:
            df = pd.read_csv(csv_file)
            st.dataframe(df.head())
            if 'target' not in df.columns:
                st.error("CSV must have a 'target' column.")
            else:
                tabular_features = df.drop(columns=['target']).select_dtypes(include=np.number).columns.tolist()
                X_tab = df[tabular_features].values
                y = df['target'].values

                image_features = None
                if zip_file:
                    import zipfile
                    from PIL import Image
                    import torchvision.transforms as transforms
                    from torchvision import models
                    import io

                    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
                    resnet.eval()

                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

                    with zipfile.ZipFile(zip_file, 'r') as z:
                        image_files = [f for f in z.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if len(image_files) != len(df):
                            st.warning(f"Number of images ({len(image_files)}) does not match rows in CSV ({len(df)}). Proceeding with tabular only.")
                        else:
                            st.info(f"Extracting features from {len(image_files)} images...")
                            image_feats = []
                            for img_name in image_files:
                                img_data = z.read(img_name)
                                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                                input_tensor = preprocess(img).unsqueeze(0)
                                with torch.no_grad():
                                    feat = resnet(input_tensor).squeeze().numpy()
                                image_feats.append(feat)
                            image_features = np.array(image_feats)
                            st.write(f"Image feature shape: {image_features.shape}")

                if image_features is not None:
                    X_combined = np.hstack([X_tab, image_features])
                else:
                    X_combined = X_tab

                if st.button("Train Multimodal Model"):
                    # Simplified training
                    from sklearn.model_selection import train_test_split
                    scaler = StandardScaler().fit(X_combined)
                    X_scaled = scaler.transform(X_combined)
                    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

                    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
                    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1,1)
                    X_te_t = torch.tensor(X_te, dtype=torch.float32)

                    class FusionNet(nn.Module):
                        def __init__(self, input_dim):
                            super().__init__()
                            self.net = nn.Sequential(
                                nn.Linear(input_dim, 32),
                                nn.ReLU(),
                                nn.Linear(32, 1),
                                nn.Sigmoid()
                            )
                        def forward(self, x):
                            return self.net(x)

                    model = FusionNet(X_tr.shape[1])
                    opt = torch.optim.Adam(model.parameters(), lr=0.001)
                    loss_fn = nn.BCELoss()

                    progress = st.progress(0)
                    for epoch in range(50):
                        opt.zero_grad()
                        loss = loss_fn(model(X_tr_t), y_tr_t)
                        loss.backward()
                        opt.step()
                        progress.progress((epoch+1)/50)

                    model.eval()
                    with torch.no_grad():
                        probs = model(X_te_t).numpy().flatten()
                    auc = roc_auc_score(y_te, probs)
                    st.success(f"Model trained. Test AUC: {auc:.3f}")
                    st.info("This is a simplified demo. Full multimodal training requires more data and careful validation.")
        else:
            st.info("Upload tabular data to start multimodal fusion.")

def deep_biomarker_analysis():
    st.title("🧬 Deep Biomarker Analyzer")
    st.markdown("""
    Upload a CSV with **numeric biomarker columns** and a binary **GROUP** column (0/1).  
    The tool will:
    1. Analyze each biomarker statistically (mean, SD, N, t‑test, effect size).
    2. Compute pairwise correlations.
    3. Train a neural network to discover hidden patterns.
    4. Generate an **explainable summary** of the findings.
    """)

    uploaded = st.file_uploader("Upload biomarker data CSV", type="csv", key="dba_csv")
    if uploaded is None:
        st.info("Please upload a CSV file.")
        return

    df = pd.read_csv(uploaded)
    if 'GROUP' not in df.columns:
        st.error("CSV must contain a 'GROUP' column with 0/1 values.")
        return

    groups = sorted(df['GROUP'].unique())
    if len(groups) != 2:
        st.error("Group column must have exactly 2 distinct values (0 and 1).")
        return

    biomarker_cols = [c for c in df.columns if c != 'GROUP' and pd.api.types.is_numeric_dtype(df[c])]
    if not biomarker_cols:
        st.error("No numeric biomarker columns found.")
        return

    st.subheader("1️⃣ Biomarker‑by‑Biomarker Analysis")

    results = []
    for col in biomarker_cols:
        g0 = df[df['GROUP'] == groups[0]][col].dropna()
        g1 = df[df['GROUP'] == groups[1]][col].dropna()

        mean0, sd0, n0 = g0.mean(), g0.std(), len(g0)
        mean1, sd1, n1 = g1.mean(), g1.std(), len(g1)

        t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False)

        pooled_sd = np.sqrt(((n0-1)*sd0**2 + (n1-1)*sd1**2) / (n0+n1-2))
        cohens_d = (mean1 - mean0) / pooled_sd if pooled_sd > 0 else 0.0

        sig = "significant" if p_val < 0.05 else "not significant"
        direction = "higher" if mean1 > mean0 else "lower"
        effect_size = "small" if abs(cohens_d) < 0.5 else ("moderate" if abs(cohens_d) < 0.8 else "large")

        explanation = (
            f"**{col}**: {sig} difference (p = {p_val:.4f}). "
            f"Group {groups[1]} shows {direction} values (mean = {mean1:.2f}) compared to Group {groups[0]} (mean = {mean0:.2f}). "
            f"Effect size (Cohen's d) = {cohens_d:.2f} ({effect_size})."
        )
        results.append({
            'Biomarker': col,
            'Mean_Group0': mean0,
            'SD_Group0': sd0,
            'N_Group0': n0,
            'Mean_Group1': mean1,
            'SD_Group1': sd1,
            'N_Group1': n1,
            'p_value': p_val,
            'Cohens_d': cohens_d,
            'Explanation': explanation
        })

        with st.expander(f"{col} (p = {p_val:.4f})"):
            st.markdown(explanation)
            fig = go.Figure()
            fig.add_trace(go.Box(y=g0, name=f"Group {groups[0]}"))
            fig.add_trace(go.Box(y=g1, name=f"Group {groups[1]}"))
            fig.update_layout(title=f"{col} distribution by group", height=400)
            st.plotly_chart(fig, use_container_width=True)

    results_df = pd.DataFrame(results)

    st.subheader("2️⃣ Correlation Matrix")
    corr_matrix = df[biomarker_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                         color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig_corr.update_layout(title="Biomarker Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    strong_pairs = []
    for i in range(len(biomarker_cols)):
        for j in range(i+1, len(biomarker_cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.7:
                strong_pairs.append((biomarker_cols[i], biomarker_cols[j], r))

    if strong_pairs:
        st.markdown("### Strong Correlations (|r| > 0.7)")
        for pair in strong_pairs:
            st.write(f"- **{pair[0]}** and **{pair[1]}**: r = {pair[2]:.2f}")
    else:
        st.info("No strong correlations (|r| > 0.7) found among biomarkers.")

    st.subheader("3️⃣ Deep Learning Pattern Discovery")
    st.markdown("Training a small autoencoder to uncover hidden structure…")

    X = df[biomarker_cols].fillna(df[biomarker_cols].median()).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class Autoencoder(nn.Module):
        def __init__(self, input_dim, encoding_dim=2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, encoding_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 16),
                nn.ReLU(),
                nn.Linear(16, input_dim)
            )
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded, encoded

    input_dim = X_scaled.shape[1]
    model = Autoencoder(input_dim, encoding_dim=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    epochs = 100
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed, encoded = model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()
        progress_bar.progress((epoch+1)/epochs)

    st.success(f"Autoencoder trained. Reconstruction loss: {loss.item():.4f}")

    with torch.no_grad():
        _, encoded = model(X_tensor)
    embeddings = encoded.numpy()

    df_embed = pd.DataFrame(embeddings, columns=['Dim1', 'Dim2'])
    df_embed['Group'] = df['GROUP'].values
    fig_embed = px.scatter(df_embed, x='Dim1', y='Dim2', color='Group',
                           title="Autoencoder Latent Space (2D)")
    st.plotly_chart(fig_embed, use_container_width=True)

    lr = LogisticRegression(max_iter=1000).fit(X_scaled, df['GROUP'])
    importance = np.abs(lr.coef_[0])
    importance_df = pd.DataFrame({
        'Biomarker': biomarker_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    fig_imp = px.bar(importance_df, x='Importance', y='Biomarker', orientation='h',
                     title="Feature Importance for Group Separation (Logistic Regression)")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("4️⃣ Explainable Summary")
    st.markdown("Based on the analyses, here is an automatic summary:")

    sig_biomarkers = results_df[results_df['p_value'] < 0.05]
    if not sig_biomarkers.empty:
        summary = f"**Significant differences** were found in {len(sig_biomarkers)} biomarkers: "
        summary += ", ".join(sig_biomarkers['Biomarker'].tolist()) + ".\n\n"
    else:
        summary = "**No biomarkers** showed statistically significant differences between groups.\n\n"

    for _, row in sig_biomarkers.iterrows():
        summary += f"- **{row['Biomarker']}**: {row['Explanation']}\n"
    summary += "\n"

    if strong_pairs:
        summary += f"**Strong correlations** were observed between: "
        summary += "; ".join([f"{p[0]} & {p[1]} (r={p[2]:.2f})" for p in strong_pairs])
        summary += ".\n"
    else:
        summary += "**No strong correlations** were found.\n"

    summary += "\n**Deep learning latent space** shows the separation between groups. "
    summary += "The most important biomarkers for classification were: "
    top_import = importance_df.sort_values('Importance', ascending=False).head(3)['Biomarker'].tolist()
    summary += ", ".join(top_import) + "."

    st.markdown(summary)

    st.download_button("Download Summary (TXT)", summary, "biomarker_summary.txt", "text/plain")

def sign_up():
    st.title("👤 Sign Up")
    st.markdown("Create a free account to save your analyses and access premium features (coming soon).")

    with st.form("signup_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        institution = st.text_input("Institution / Company")
        role = st.selectbox("Role", ["Researcher", "Clinician", "Student", "Industry Professional", "Other"])
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Sign Up")

        if submit:
            if not name or not email or not password:
                st.error("Please fill in all required fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                # Save to CSV (demo)
                user_data = pd.DataFrame({
                    'Name': [name],
                    'Email': [email],
                    'Institution': [institution],
                    'Role': [role],
                    'Timestamp': [pd.Timestamp.now()]
                })
                try:
                    existing = pd.read_csv('users.csv')
                    user_data = pd.concat([existing, user_data], ignore_index=True)
                except FileNotFoundError:
                    pass
                user_data.to_csv('users.csv', index=False)
                st.success(f"Thank you {name}! Your account has been created (demo only).")
                st.balloons()

# ==========================================
# NAVIGATION SETUP
# ==========================================
pages = [
    st.Page(home, title="Home", icon="🏠"),
    st.Page(in_vivo, title="In Vivo", icon="🧪"),
    st.Page(in_vitro, title="In Vitro", icon="🧫"),
    st.Page(clinical, title="Clinical", icon="🩺"),
    st.Page(stats_graphs, title="Stats & Graph Maker", icon="📊"),
    st.Page(deep_learning, title="Deep Learning & Multimodal AI", icon="🧠"),
    st.Page(deep_biomarker_analysis, title="Deep Biomarker Analyzer", icon="🧬"),
    st.Page(sign_up, title="Sign Up", icon="👤"),
]

nav = st.navigation(pages)
nav.run()
