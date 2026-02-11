import streamlit as st
import pandas as pd
import pickle as pk
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="AI Loan Genius - Smart Lending Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Models and Feature Names
@st.cache_resource
def load_models():
    try:
        model = pk.load(open('model.pkl', 'rb'))
        scaler = pk.load(open('scaler.pkl', 'rb'))
        try:
            all_models = pk.load(open('all_models.pkl', 'rb'))
            feature_names = pk.load(open('feature_names.pkl', 'rb'))
        except:
            all_models = None
            feature_names = None
        return model, scaler, all_models, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, scaler, all_models, feature_names = load_models()

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Simplified but Beautiful CSS - FIXED Z-INDEX ISSUES
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Background */
        .stApp {
            background: linear-gradient(-45deg, #0a0e27, #16213e, #1a1a3e, #0a0e27);
            background-size: 400% 400%;
            animation: gradientShift 20s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Main content container - FIXED */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
            position: relative;
            z-index: 1;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #00f5ff !important;
            font-weight: 700 !important;
        }
        
        /* Input fields */
        .stNumberInput input {
            background: rgba(20, 25, 45, 0.95) !important;
            border: 2px solid rgba(0, 245, 255, 0.3) !important;
            border-radius: 12px !important;
            color: #ffffff !important;
            font-size: 16px !important;
            padding: 12px !important;
        }
        
        .stNumberInput input:focus {
            border: 2px solid #00f5ff !important;
            box-shadow: 0 0 0 3px rgba(0, 245, 255, 0.2) !important;
            outline: none !important;
        }
        
        .stNumberInput label, .stSelectbox label {
            color: #00f5ff !important;
            font-size: 14px !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
        }
        
        /* Fix text color */
        input[type="number"] {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        
        /* SELECTBOX - Complete Fix */
        .stSelectbox > div > div {
            background: rgba(20, 25, 45, 0.95) !important;
            border: 2px solid rgba(0, 245, 255, 0.3) !important;
            border-radius: 12px !important;
        }
        
        .stSelectbox > div > div:hover {
            border-color: #00f5ff !important;
        }
        
        /* Selected value in dropdown */
        .stSelectbox > div > div > div {
            color: #ffffff !important;
        }
        
        /* Dropdown menu when opened */
        [data-baseweb="popover"] {
            background: #1a1f3a !important;
            border: 2px solid #00f5ff !important;
            border-radius: 12px !important;
        }
        
        /* Dropdown options */
        [data-baseweb="menu"] {
            background: #1a1f3a !important;
        }
        
        [role="option"] {
            background: #1a1f3a !important;
            color: #ffffff !important;
            padding: 12px 16px !important;
        }
        
        [role="option"]:hover {
            background: #2a3f5a !important;
            color: #00f5ff !important;
        }
        
        /* Selected option highlighting */
        [aria-selected="true"] {
            background: #2a3f5a !important;
            color: #00f5ff !important;
        }
        
        /* Dropdown list container */
        ul[role="listbox"] {
            background: #1a1f3a !important;
            border: 2px solid #00f5ff !important;
            border-radius: 12px !important;
        }
        
        ul[role="listbox"] li {
            background: #1a1f3a !important;
            color: #ffffff !important;
            padding: 12px 16px !important;
        }
        
        ul[role="listbox"] li:hover {
            background: #2a3f5a !important;
            color: #00f5ff !important;
        }
        
        /* Button */
        .stButton > button {
            background: linear-gradient(135deg, #00f5ff 0%, #0090ff 100%) !important;
            color: #000000 !important;
            padding: 18px 40px !important;
            border-radius: 50px !important;
            font-size: 20px !important;
            font-weight: 800 !important;
            border: none !important;
            width: 100% !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            box-shadow: 0 0 30px rgba(0, 245, 255, 0.5) !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            transform: scale(1.02) !important;
            box-shadow: 0 0 50px rgba(0, 245, 255, 0.8) !important;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0a0e27, #16213e, #0f1b2e) !important;
        }
        
        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* Sidebar spacing - prevent overlap */
        section[data-testid="stSidebar"] > div {
            padding: 1rem !important;
        }
        
        section[data-testid="stSidebar"] .element-container {
            margin-bottom: 0.5rem !important;
        }
        
        section[data-testid="stSidebar"] hr {
            margin: 1.5rem 0 !important;
            border-color: rgba(0, 245, 255, 0.3) !important;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #00f5ff !important;
            font-size: 24px !important;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Info box */
        .stAlert {
            background: rgba(0, 245, 255, 0.1) !important;
            border: 1px solid rgba(0, 245, 255, 0.3) !important;
            color: #ffffff !important;
        }
        
        /* Code blocks in expanders */
        pre, code {
            background: rgba(10, 14, 39, 0.8) !important;
            color: #00f5ff !important;
            padding: 8px !important;
            border-radius: 6px !important;
            border: 1px solid rgba(0, 245, 255, 0.2) !important;
        }
        
        /* Dataframe styling - BRIGHT AND VISIBLE */
        .dataframe {
            background: rgba(15, 20, 40, 1) !important;
            color: #ffffff !important;
            border: 2px solid rgba(0, 245, 255, 0.3) !important;
        }
        
        .dataframe th {
            background: rgba(0, 120, 130, 0.9) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            padding: 10px !important;
            border: 1px solid rgba(0, 245, 255, 0.4) !important;
        }
        
        .dataframe td {
            color: #ffffff !important;
            background: rgba(20, 25, 45, 0.9) !important;
            border: 1px solid rgba(0, 245, 255, 0.2) !important;
            padding: 8px !important;
            font-weight: 500 !important;
        }
        
        /* Dataframe index column */
        .dataframe tbody tr th {
            background: rgba(0, 120, 130, 0.7) !important;
            color: #00f5ff !important;
            font-weight: 700 !important;
        }
        
        /* Dataframe container */
        [data-testid="stDataFrame"] {
            background: rgba(15, 20, 40, 1) !important;
            padding: 10px !important;
            border-radius: 8px !important;
        }
        
        /* All table elements */
        table {
            color: #ffffff !important;
        }
        
        table th, table td {
            color: #ffffff !important;
        }
        
        /* Text elements inside main content */
        .stMarkdown {
            color: rgba(255, 255, 255, 0.9) !important;
        }
        
        /* Fix "Expected Features" text visibility */
        .main h3, .main h4, .main h5 {
            color: #00f5ff !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background: rgba(26, 31, 58, 1) !important;
            color: #00f5ff !important;
            border-radius: 8px !important;
            margin-bottom: 10px !important;
            border: 2px solid rgba(0, 245, 255, 0.5) !important;
            padding: 14px 16px !important;
            font-weight: 600 !important;
            font-size: 14px !important;
            position: relative !important;
            z-index: 1 !important;
        }
        
        /* Expander arrow icon */
        .streamlit-expanderHeader svg {
            color: #00f5ff !important;
            min-width: 20px !important;
        }
        
        /* Expander content - SOLID background */
        .streamlit-expanderContent {
            background: rgba(15, 20, 40, 1) !important;
            border: 2px solid rgba(0, 245, 255, 0.3) !important;
            border-top: none !important;
            border-radius: 0 0 8px 8px !important;
            padding: 16px !important;
            margin-top: -8px !important;
            margin-bottom: 24px !important;
            position: relative !important;
            z-index: 0 !important;
        }
        
        /* Fix expander spacing */
        [data-testid="stExpander"] {
            margin-bottom: 24px !important;
            background: transparent !important;
            position: relative !important;
            clear: both !important;
        }
        
        /* Expander text - ensure visibility */
        .streamlit-expanderContent p,
        .streamlit-expanderContent div,
        .streamlit-expanderContent span,
        .streamlit-expanderContent strong {
            color: #ffffff !important;
            line-height: 1.8 !important;
        }
        
        /* Sidebar expander specific - MORE SPACING */
        section[data-testid="stSidebar"] [data-testid="stExpander"] {
            margin-bottom: 20px !important;
            display: block !important;
            clear: both !important;
        }
        
        section[data-testid="stSidebar"] .streamlit-expanderHeader {
            font-size: 13px !important;
            padding: 12px 14px !important;
            background: rgba(26, 31, 58, 1) !important;
            margin-bottom: 0 !important;
        }
        
        section[data-testid="stSidebar"] .streamlit-expanderContent {
            font-size: 13px !important;
            padding: 14px !important;
            background: rgba(15, 20, 40, 1) !important;
            margin-bottom: 20px !important;
        }
        
        /* Fix text inside expanders */
        section[data-testid="stSidebar"] .streamlit-expanderContent * {
            color: #ffffff !important;
        }
        
        /* Main page expander text visibility */
        .main [data-testid="stExpander"] .streamlit-expanderContent {
            background: rgba(15, 20, 40, 1) !important;
        }
        
        .main [data-testid="stExpander"] p,
        .main [data-testid="stExpander"] pre,
        .main [data-testid="stExpander"] code {
            color: #ffffff !important;
        }
        
        /* Force spacing between sidebar elements */
        section[data-testid="stSidebar"] .element-container {
            margin-bottom: 1rem !important;
        }
        
        /* General text color */
        p, span, div {
            color: rgba(255, 255, 255, 0.9);
        }
    </style>
""", unsafe_allow_html=True)

# Feature calculation functions - CORRECTED TO MATCH MODEL TRAINING
def calculate_all_features(no_of_dep, grad_s, emp_s, annual_income, loan_amount, loan_dur, cibil, assets):
    """
    Calculate all features exactly as done during model training.
    This must match the feature engineering in model.py lines 70-108.
    """
    # Base features (matching model.py)
    features = {
        'no_of_dependents': no_of_dep,
        'education': grad_s,
        'self_employed': emp_s,
        'income_annum': annual_income,
        'loan_amount': loan_amount,
        'loan_term': loan_dur,
        'cibil_score': cibil,
        'total_assets': assets  # Changed from 'Assets' to 'total_assets'
    }
    
    # Engineered features - EXACTLY as in model.py lines 79-92
    
    # 1. high_income (binary: above median income)
    # Using a reasonable median estimate (can be adjusted)
    median_income_estimate = 5000000  # Adjust based on your training data
    features['high_income'] = 1 if annual_income > median_income_estimate else 0
    
    # 2. asset_level (categorical: 0, 1, 2)
    if assets <= 1000000:
        features['asset_level'] = 0
    elif assets <= 5000000:
        features['asset_level'] = 1
    else:
        features['asset_level'] = 2
    
    # 3. employment_risk (1 if self-employed, 0 otherwise)
    features['employment_risk'] = emp_s
    
    # 4. family_size (categorical based on dependents)
    if no_of_dep <= 1:
        features['family_size'] = 0
    elif no_of_dep <= 3:
        features['family_size'] = 1
    else:
        features['family_size'] = 2
    
    # Calculate display metrics (for UI only, not used in prediction)
    dti_ratio = (loan_amount / loan_dur) / (annual_income / 12) * 100 if loan_dur > 0 and annual_income > 0 else 0
    if pd.isna(dti_ratio) or np.isinf(dti_ratio):
        dti_ratio = 0
    
    ltv_ratio = (loan_amount / assets) * 100 if assets > 0 else 100
    if pd.isna(ltv_ratio) or np.isinf(ltv_ratio):
        ltv_ratio = 100
    
    return features, dti_ratio, ltv_ratio

def calculate_risk_score(cibil, dti, ltv, dependents):
    """Calculate a simple risk score for display purposes"""
    cibil_score = ((cibil - 300) / 600) * 40
    dti_score = max(0, (50 - dti) / 50) * 25
    ltv_score = max(0, (80 - ltv) / 80) * 20
    dep_score = max(0, (5 - dependents) / 5) * 15
    risk_score = 100 - (cibil_score + dti_score + ltv_score + dep_score)
    return max(0, min(100, risk_score))

def get_risk_category(risk_score):
    """Categorize risk score"""
    if risk_score <= 20:
        return "🟢 Excellent"
    elif risk_score <= 35:
        return "🟡 Good"
    elif risk_score <= 50:
        return "🟠 Moderate"
    elif risk_score <= 70:
        return "🔴 High"
    else:
        return "⛔ Very High"

# Header
st.title("🎯 AI LOAN GENIUS")
st.markdown("### *Intelligent Lending • Instant Decisions • Personalized Insights*")
st.markdown("---")

# Model info
if model and feature_names:
    model_name = type(model).__name__
    st.info(f"🤖 **Active Model:** {model_name} | **Features:** {len(feature_names)}")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    if all_models:
        model_names = ['Best Model (Gradient Boosting)'] + list(all_models.keys())
        selected_model = st.selectbox("🤖 Select Model", model_names)
    
    show_debug = st.checkbox("🔍 Debug Mode", value=False)
    show_feature_importance = st.checkbox("📊 Feature Details", value=False)
    
    st.markdown("---")
    st.metric("🕐 Time", datetime.now().strftime("%H:%M"))
    if model:
        st.metric("🤖 Model", type(model).__name__)
    
    # Debug Information Section
    if show_debug:
        st.markdown("---")
        st.markdown("### 🐛 Debug Info")
        st.markdown("<br>", unsafe_allow_html=True)  # Extra spacing
        
        # Model Details
        with st.expander("🤖 Model Details", expanded=False):
            if model:
                st.markdown(f"**Type:** `{type(model).__name__}`")
                if hasattr(model, 'n_estimators'):
                    st.markdown(f"**Estimators:** `{model.n_estimators}`")
                if hasattr(model, 'max_depth'):
                    st.markdown(f"**Max Depth:** `{model.max_depth}`")
                if hasattr(model, 'learning_rate'):
                    st.markdown(f"**Learning Rate:** `{model.learning_rate}`")
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Feature Names
        with st.expander("📋 Feature Names", expanded=False):
            if feature_names:
                st.markdown(f"**Total Features:** `{len(feature_names)}`")
                st.markdown("")
                for i, fname in enumerate(feature_names, 1):
                    st.markdown(f"`{i}.` **{fname}**")
            else:
                st.warning("Feature names not loaded")
        
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Scaler Info
        with st.expander("⚖️ Scaler Info", expanded=False):
            if scaler:
                st.markdown(f"**Type:** `{type(scaler).__name__}`")
                if hasattr(scaler, 'mean_'):
                    st.markdown(f"**Features Scaled:** `{len(scaler.mean_)}`")
            else:
                st.warning("Scaler not loaded")
    
    # Feature importance
    if show_feature_importance and hasattr(model, 'feature_importances_') and feature_names:
        st.markdown("---")
        st.markdown("### 🎯 Top Features")
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-5:][::-1]
        for idx in top_indices:
            st.text(f"{feature_names[idx][:20]}: {importances[idx]:.1%}")
    
    # History
    if st.session_state.prediction_history:
        st.markdown("---")
        st.markdown("### 📜 Prediction History")
        approved = sum(1 for p in st.session_state.prediction_history if p == 1)
        total = len(st.session_state.prediction_history)
        st.metric("Approval Rate", f"{approved/total*100:.0f}%")
        st.metric("Total Predictions", total)
        
        if show_debug:
            with st.expander("📊 History Details"):
                st.write(f"Approved: {approved}")
                st.write(f"Rejected: {total - approved}")
                st.write(f"Last 5: {st.session_state.prediction_history[-5:]}")
    
    # System Info
    if show_debug:
        st.markdown("---")
        with st.expander("💻 System Info"):
            st.write(f"**Pandas:** {pd.__version__}")
            st.write(f"**NumPy:** {np.__version__}")
            st.write(f"**Streamlit:** {st.__version__}")

# Main Content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("👤 PERSONAL INFORMATION")
    grad = st.selectbox('🎓 Education', ['Graduate', 'Not Graduate'])
    self_emp = st.selectbox('💼 Employment', ['No', 'Yes'])
    no_of_dep = st.number_input('👨‍👩‍👧‍👦 Dependents', 0, 10, 0)

with col2:
    st.subheader("💰 FINANCIAL PROFILE")
    Annual_Income = st.number_input('💵 Annual Income (₹)', 0, 100000000, 3000000, step=100000)
    Assets = st.number_input('🏡 Total Assets (₹)', 0, 500000000, 5000000, step=100000)
    Cibil = st.number_input('📊 CIBIL Score', 300, 900, 650)

st.subheader("🏦 LOAN REQUIREMENTS")
col3, col4 = st.columns(2)

with col3:
    Loan_Amount = st.number_input('💳 Loan Amount (₹)', 0, 500000000, 2000000, step=100000)

with col4:
    Loan_Dur = st.number_input('⏳ Duration (Years)', 1, 30, 10)

# Real-time metrics
if Annual_Income > 0 and Loan_Amount > 0 and Loan_Dur > 0:
    grad_s = 1 if grad == 'Graduate' else 0
    emp_s = 1 if self_emp == 'Yes' else 0
    
    features, dti_ratio, ltv_ratio = calculate_all_features(
        no_of_dep, grad_s, emp_s, Annual_Income, 
        Loan_Amount, Loan_Dur, Cibil, Assets
    )
    
    risk_score = calculate_risk_score(Cibil, dti_ratio, ltv_ratio, no_of_dep)
    monthly_payment = Loan_Amount / (Loan_Dur * 12)
    
    st.subheader("📊 LIVE METRICS")
    metric_cols = st.columns(5)
    
    metric_cols[0].metric("Monthly EMI", f"₹{monthly_payment:,.0f}")
    metric_cols[1].metric("DTI Ratio", f"{dti_ratio:.1f}%")
    metric_cols[2].metric("LTV Ratio", f"{ltv_ratio:.1f}%")
    metric_cols[3].metric("Risk Score", f"{risk_score:.0f}")
    metric_cols[4].metric("Coverage", f"{(Assets/Loan_Amount):.1f}x" if Loan_Amount > 0 else "N/A")

st.markdown("---")

# Prediction Button
if st.button("🚀 ANALYZE ELIGIBILITY"):
    if not model or not scaler:
        st.error("❌ Model files not found!")
    else:
        with st.spinner('🤖 AI analyzing your application...'):
            grad_s = 1 if grad == 'Graduate' else 0
            emp_s = 1 if self_emp == 'Yes' else 0
            
            # Calculate all features
            features, dti, ltv = calculate_all_features(
                no_of_dep, grad_s, emp_s, Annual_Income, 
                Loan_Amount, Loan_Dur, Cibil, Assets
            )
            
            if feature_names:
                # Create DataFrame with features in the exact order expected by the model
                pred_data = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)
            else:
                st.error("Feature names not found.")
                st.stop()
            
            # Debug
            if show_debug:
                with st.expander("🔍 Debug Information", expanded=False):
                    st.markdown("### Expected Features:")
                    st.code('\n'.join(feature_names), language=None)
                    
                    st.markdown("")
                    st.markdown("### Encodings:")
                    st.markdown(f"- **Education:** `{grad}` → `{grad_s}`")
                    st.markdown(f"- **Self-Employed:** `{self_emp}` → `{emp_s}`")
                    
                    st.markdown("")
                    st.markdown("### Calculated Features:")
                    st.dataframe(pred_data.T, width=True)
            
            # Predict
            try:
                pred_data_scaled = scaler.transform(pred_data)
                prediction = model.predict(pred_data_scaled)
                prediction_proba = model.predict_proba(pred_data_scaled) if hasattr(model, 'predict_proba') else None
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
                if show_debug:
                    st.exception(e)
                st.stop()
            
            st.session_state.prediction_history.append(prediction[0])
            
            risk_score = calculate_risk_score(Cibil, dti, ltv, no_of_dep)
            risk_cat = get_risk_category(risk_score)
            
            st.markdown("---")
            
            # Result
            if prediction[0] == 1:
                st.success("# ✅ LOAN APPROVED! 🎉")
                if prediction_proba is not None:
                    confidence = prediction_proba[0][1] * 100
                    st.info(f"**Confidence:** {confidence:.1f}%")
            else:
                st.error("# ❌ LOAN NOT APPROVED")
                if prediction_proba is not None:
                    confidence = prediction_proba[0][0] * 100
                    st.warning(f"**Confidence:** {confidence:.1f}%")
            
            st.info(f"**Risk Category:** {risk_cat}")
            
            # Insights
            st.markdown("### 🧠 AI Insights")
            
            if prediction[0] == 1:
                st.markdown("""
                **Your application was approved based on:**
                - ✅ Strong CIBIL score
                - ✅ Healthy debt-to-income ratio
                - ✅ Adequate asset coverage
                """)
            else:
                st.markdown("**Improvement Suggestions:**")
                
                if Cibil < 700:
                    st.markdown(f"- 📊 **Improve CIBIL Score:** Current {Cibil}, target 750+")
                
                if dti > 35:
                    st.markdown(f"- 💳 **Reduce DTI Ratio:** Current {dti:.1f}%, target <35%")
                
                if ltv > 70:
                    st.markdown(f"- 🏠 **Increase Assets:** Current LTV {ltv:.1f}%, target <70%")
                
                if no_of_dep > 3:
                    st.markdown(f"- 👨‍👩‍👧 **High Dependents:** {no_of_dep} dependents may affect approval")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.5); padding: 20px;'>
    🔒 Bank-Grade Security • 🤖 Advanced AI • 📊 Real-Time Analytics<br>
    © 2026 AI Loan Genius - Powered by Gradient Boosting ML
</div>
""", unsafe_allow_html=True)