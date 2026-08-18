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
    st.markdown("Patient cohort analysis, biomarker discovery, and risk prediction.")

    # Reuse a simplified version of the original liver analysis
    st.subheader("Biomarker Discovery and Classification")
    uploaded = st.file_uploader("Upload clinical CSV (must contain 'GROUP' 0/1)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success("Data loaded.")
        st.dataframe(df.head())

        if 'GROUP' in df.columns:
            # Derived indices
            if {'AST', 'ALT'}.issubset(df.columns):
                df['AST_ALT_Ratio'] = df['AST'] / df['ALT'].replace(0, np.nan)
            if {'TG', 'HDL-C'}.issubset(df.columns):
                df['TG_HDL_Ratio'] = df['TG'] / df['HDL-C'].replace(0, np.nan)

            biomarkers = [c for c in df.select_dtypes(include=np.number).columns if c != 'GROUP']
            corr = df[biomarkers].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Feature Importance using Logistic Regression**")
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
            X = df[biomarkers].fillna(df[biomarkers].median())
            y = df['GROUP']
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr)
            lr = LogisticRegression(max_iter=1000).fit(X_tr_s, y_tr)
            importance = pd.DataFrame({
                'Biomarker': biomarkers,
                'Coefficient': np.abs(lr.coef_[0])
            }).sort_values('Coefficient', ascending=True)
            fig = px.bar(importance, x='Coefficient', y='Biomarker', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("CSV must contain a 'GROUP' column (0/1) for classification.")
    else:
        st.info("Upload clinical data to perform biomarker analysis.")

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

    tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "📄 Text Mining", "🔗 Multimodal Fusion"])

    with tab1:
        st.subheader("Image Classification (Placeholder)")
        st.markdown("Upload an image (e.g., histology, MRI) for classification using a pre‑trained model.")
        uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded:
            from PIL import Image
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.info("Model integration coming soon. Currently displaying image only.")
        else:
            st.info("Upload an image to test the AI module.")

    with tab2:
        st.subheader("Text Mining (PubMed Abstracts)")
        st.markdown("Paste text or upload a PDF to extract key information using NLP.")
        text_input = st.text_area("Enter text for analysis", height=200)
        if st.button("Analyze Text"):
            if text_input:
                # Simple word frequency placeholder
                words = text_input.lower().split()
                freq = pd.Series(words).value_counts().head(20)
                fig = px.bar(freq, orientation='h')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter some text.")

    with tab3:
        st.subheader("Multimodal Fusion Model")
        st.markdown("Combine tabular clinical data with imaging or genomic features.")
        st.info("This module will allow you to upload multiple data types and train a joint neural network. Coming soon.")

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
