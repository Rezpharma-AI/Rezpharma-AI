import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.optimize import curve_fit
import io
import base64

# ==========================================
# PAGE CONFIG & GLOBAL STYLES
# ==========================================
st.set_page_config(
    page_title="RezPharma AI Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
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
    .formal-box h1 { color: white; }
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
</style>
""", unsafe_allow_html=True)

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

            # Basic statistics by group
            group_col = st.selectbox("Grouping column", df.columns)
            value_col = st.selectbox("Value column", df.select_dtypes(include=np.number).columns)

            if group_col and value_col:
                groups = df[group_col].unique()
                if len(groups) >= 2:
                    # T-test or ANOVA
                    if len(groups) == 2:
                        g1 = df[df[group_col] == groups[0]][value_col].dropna()
                        g2 = df[df[group_col] == groups[1]][value_col].dropna()
                        t_stat, p_val = stats.ttest_ind(g1, g2)
                        st.markdown(f"**T‑test p‑value:** {p_val:.4f}")
                    else:
                        # One‑way ANOVA
                        groups_data = [df[df[group_col] == g][value_col].dropna() for g in groups]
                        f_stat, p_val = stats.f_oneway(*groups_data)
                        st.markdown(f"**ANOVA p‑value:** {p_val:.4f}")

                    # Plot
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
            # Simple AUC calculation using trapezoidal rule
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
        st.markdown("Placeholder for dose‑response toxicity curves and survival analysis.")
        st.info("This module will be expanded to include Kaplan‑Meier survival plots and dose‑toxicity modelling.")

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

                # Fit Hill equation
                try:
                    p0 = [np.min(y), np.max(y), np.median(x), 1]
                    popt, _ = curve_fit(hill_equation, x, y, p0=p0, maxfev=5000)
                    bottom, top, ic50, hill = popt
                    st.success(f"Fitted IC50: **{ic50:.3f}** (concentration units)")
                    st.write(f"Hill slope: {hill:.3f}")

                    # Generate curve
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
        st.subheader("Plate Reader Data Normalization")
        st.markdown("Upload raw plate reader data (e.g., 96‑well format) or a CSV with well IDs.")
        st.info("This module is under development. Please check back later.")

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

        # Derived indices
        if {'AST', 'ALT'}.issubset(df.columns):
            df['AST_ALT_Ratio'] = df['AST'] / df['ALT'].replace(0, np.nan)
        if {'TG', 'HDL-C'}.issubset(df.columns):
            df['TG_HDL_Ratio'] = df['TG'] / df['HDL-C'].replace(0, np.nan)
        if {'AST', 'PLT'}.issubset(df.columns):
            df['APRI'] = ((df['AST'] / 40) / df['PLT'].replace(0, np.nan)) * 100

        # Identify numeric biomarkers (exclude GROUP)
        biomarkers = [c for c in df.select_dtypes(include=np.number).columns if c != 'GROUP']

        # Show correlation matrix
        st.subheader("Correlation Matrix")
        corr = df[biomarkers].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)

        # ----- Train AI model -----
        st.subheader("Train AI Model (Logistic Regression + PyTorch MLP)")
        if st.button("🚀 Train Model"):
            import torch
            import torch.nn as nn
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score, roc_curve

            X = df[biomarkers].fillna(df[biomarkers].median()).values
            y = df['GROUP'].values

            # Train/test split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            if len(np.unique(y_te)) < 2:
                st.error("Test set contains only one class. Cannot compute ROC.")
                return

            # Scale
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Logistic Regression
            lr = LogisticRegression(max_iter=1000).fit(X_tr_s, y_tr)
            lr_probs = lr.predict_proba(X_te_s)[:, 1]
            lr_auc = roc_auc_score(y_te, lr_probs)

            # PyTorch MLP
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

            # Convert to tensors
            X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
            y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
            X_te_t = torch.tensor(X_te_s, dtype=torch.float32)

            model = MLP(X_tr_s.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            loss_fn = nn.BCELoss()

            # Training loop with early stopping
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

                # Validation loss (using training set as simple validation)
                model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(model(X_tr_t), y_tr_t)  # you can split training into train/val

                # Update progress
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss.item():.4f}")

                # Early stopping
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

            # ROC curves
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

            # Feature importance from logistic regression
            importance = pd.DataFrame({
                'Biomarker': biomarkers,
                'Coefficient': np.abs(lr.coef_[0])
            }).sort_values('Coefficient', ascending=True)
            fig_imp = px.bar(importance, x='Coefficient', y='Biomarker', orientation='h',
                             title="Feature Importance (Logistic Regression)")
            st.plotly_chart(fig_imp, use_container_width=True)

            # Save artifacts to session state for prediction
            st.session_state['clinical_artifacts'] = {
                'model': model,
                'scaler': scaler,
                'feature_names': biomarkers,
                'lr_model': lr
            }
            st.success("Model trained successfully! You can now use the Single Patient Prediction below.")

        # ----- Single Patient Prediction -----
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

    # Data upload
    uploaded = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded:
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.success("Data loaded.")
        st.dataframe(df.head())

        # Graph builder
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
        else:  # Histogram
            fig = px.histogram(df, x=x_axis)
        st.plotly_chart(fig, use_container_width=True)

        # Statistical tests
        st.subheader("Statistical Tests")
        test_type = st.selectbox("Select test", ["T‑test (two groups)", "ANOVA (multiple groups)", "Pearson correlation", "Chi‑square test"])

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

    # ---------------- TAB 1: IMAGE CLASSIFICATION ----------------
    with tab1:
        st.subheader("Image Classification with Pre‑trained CNN (ResNet18)")
        st.markdown("Upload an image (e.g., histology, MRI) to classify it using a ResNet18 model pre‑trained on ImageNet.")

        uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="dl_image")
        if uploaded is not None:
            from PIL import Image
            import torch
            import torchvision.transforms as transforms
            from torchvision import models

            # Load image
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Load pre-trained ResNet18
            try:
                model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                model.eval()

                # Preprocess
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                input_tensor = preprocess(image).unsqueeze(0)

                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)

                # Load ImageNet labels
                import requests
                labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                try:
                    labels = requests.get(labels_url).text.splitlines()
                except:
                    labels = [f"Class {i}" for i in range(1000)]

                # Show top 5 predictions
                top5_prob, top5_catid = torch.topk(probabilities, 5)
                st.markdown("### Top Predictions:")
                for i in range(5):
                    st.write(f"{labels[top5_catid[i]]}: {top5_prob[i].item()*100:.2f}%")

            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.info("Make sure 'torch' and 'torchvision' are installed (check requirements.txt).")
        else:
            st.info("Upload an image to classify it.")

    # ---------------- TAB 2: TEXT MINING ----------------
    with tab2:
        st.subheader("Text Mining & Document Analysis")
        st.markdown("Paste text or upload multiple .txt files to extract top keywords and analyze document similarity.")

        # Input method
        input_method = st.radio("Input method", ["Paste text", "Upload files"])

        documents = []
        if input_method == "Paste text":
            text_input = st.text_area("Enter text (one document per line or a single paragraph)", height=200)
            if text_input:
                # Split by newline for multiple docs, else treat as one
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

            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
            tfidf_matrix = vectorizer.fit_transform(documents)
            feature_names = vectorizer.get_feature_names_out()

            # Sum TF-IDF scores across all documents for global keywords
            global_scores = tfidf_matrix.sum(axis=0).A1
            top_indices = global_scores.argsort()[::-1][:10]
            top_keywords = [(feature_names[i], global_scores[i]) for i in top_indices]

            # Display keywords as bar chart
            if top_keywords:
                kw_df = pd.DataFrame(top_keywords, columns=["Keyword", "Score"])
                fig = px.bar(kw_df, x="Score", y="Keyword", orientation='h')
                st.plotly_chart(fig, use_container_width=True)

            # Document similarity (cosine)
            if len(documents) > 1:
                st.subheader("Document Similarity Matrix")
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(tfidf_matrix)
                fig = px.imshow(sim_matrix, text_auto=True, aspect="auto",
                                labels=dict(x="Document", y="Document", color="Similarity"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Provide text or upload files to analyze.")

    # ---------------- TAB 3: MULTIMODAL FUSION ----------------
    with tab3:
        st.subheader("Multimodal Fusion (Concept Demo)")
        st.markdown("""
        This module demonstrates how to combine **tabular clinical data** with **image features** in a single neural network.

        **How it works:**
        1. Upload tabular data (CSV) and an image (optional).
        2. Extract image features using a pre‑trained CNN (ResNet18).
        3. Concatenate tabular features and image features.
        4. Train a joint neural network.

        **Full implementation requires a custom dataset and training loop. Below is a simplified demo.**
        """)

        # Tabular input
        uploaded_csv = st.file_uploader("Upload tabular data (CSV) with a 'target' column", type="csv", key="mm_csv")
        image_upload = st.file_uploader("Upload a representative image (optional)", type=["jpg", "png", "jpeg"], key="mm_img")

        if uploaded_csv is not None:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head())

            if 'target' in df.columns:
                # Extract tabular features
                tabular_features = df.drop(columns=['target']).select_dtypes(include=np.number).columns.tolist()
                st.write(f"Tabular features: {len(tabular_features)}")

                # If image uploaded, extract features
                image_features = None
                if image_upload is not None:
                    st.info("Extracting image features using ResNet18...")
                    from PIL import Image
                    import torch
                    import torchvision.transforms as transforms
                    from torchvision import models

                    image = Image.open(image_upload).convert("RGB")
                    st.image(image, width=200)

                    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                    # Remove final classification layer to get feature vector
                    model = torch.nn.Sequential(*list(model.children())[:-1])
                    model.eval()

                    preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_tensor = preprocess(image).unsqueeze(0)
                    with torch.no_grad():
                        image_features = model(input_tensor).squeeze().numpy()
                    st.write(f"Image feature vector length: {image_features.shape[0]}")
                else:
                    st.write("No image features (only tabular).")

                st.markdown("**Next step:** Train a neural network that takes both tabular and image features. This requires a dataset with paired tabular and image data for each sample.")
                st.info("For a full implementation, you can extend this code to loop over multiple samples, extract image features for each, and train a custom PyTorch model.")
            else:
                st.warning("CSV must contain a 'target' column.")
        else:
            st.info("Upload tabular data to start multimodal fusion demo.")

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
                # Here you would typically save to a database
                st.success(f"Thank you {name}! Your account has been created (demo only – no data stored).")
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
    st.Page(sign_up, title="Sign Up", icon="👤"),
]

nav = st.navigation(pages)
nav.run()
