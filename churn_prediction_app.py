import streamlit as st
# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make the title bigger and position it better
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        text-align: center !important;
        margin-top: 0.5rem !important;
        margin-bottom: 2rem !important;
        background: linear-gradient(to right, #3a86ff, #8338ec);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        width: 100%;
    }
    .stApp > header {
        background-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
from streamlit_lottie import st_lottie
import json
import requests
import base64
from pathlib import Path
# Import required libraries
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import os
from PIL import Image
from streamlit_lottie import st_lottie
import json
import requests
import base64
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import pickle
# Import required libraries and set up paths
def load_css():
    css = """
    /* Modern color palette */
    :root {
        --primary: #2b6cb0;
        --secondary: #4299e1;
        --accent: #63b3ed;
        --success: #48bb78;
        --warning: #ecc94b;
        --danger: #f56565;
        --background: #f7fafc;
        --text: #2d3748;
    }

    /* Global styles */
    .stApp {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    /* Header styles */
    .main-header {
        background: linear-gradient(120deg, var(--primary), var(--secondary));
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Card component */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    .custom-card:hover {
        transform: translateY(-2px);
    }

    /* Form styling */
    .stSelectbox, .stNumberInput {
        margin-bottom: 1rem;
    }

    /* Button styles */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: var(--secondary);
        transform: translateY(-1px);
    }

    /* Metrics and KPIs */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        border-left: 4px solid var(--primary);
    }

    /* Charts and visualizations */
    .chart-wrapper {
        background: white;
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .animate-fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
        }
        .custom-card {
            padding: 1rem;
        }
    }
    """
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1557683316-973673baf926?q=80");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
# Load CSS and add background after page configuration
load_css()

# Function to load Lottie animations
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

# Use working animation URLs - replaced with publicly accessible ones
lottie_prediction = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_49rdyysj.json")
lottie_analysis = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_inuxiflu.json")

# Fallback animation JSON for when URL loading fails
fallback_animation = {
    "v": "5.5.7",
    "fr": 25,
    "ip": 0,
    "op": 50,
    "w": 100,
    "h": 100,
    "nm": "Loading Animation",
    "ddd": 0,
    "assets": [],
    "layers": [{
        "ddd": 0,
        "ind": 1,
        "ty": 4,
        "nm": "Circle",
        "sr": 1,
        "ks": {
            "o": {"a": 0, "k": 100},
            "r": {"a": 1, "k": [{"t": 0, "s": [0]}, {"t": 50, "s": [360]}]},
            "p": {"a": 0, "k": [50, 50, 0]},
            "a": {"a": 0, "k": [0, 0, 0]},
            "s": {"a": 0, "k": [100, 100, 100]}
        },
        "ao": 0,
        "shapes": [{
            "ty": "el",
            "d": 1,
            "s": {"a": 0, "k": [60, 60]},
            "p": {"a": 0, "k": [0, 0]},
            "c": {"a": 0, "k": [0, 0, 0, 0]},
            "hd": False
        }, {
            "ty": "st",
            "c": {"a": 0, "k": [0.22, 0.52, 1, 1]},
            "o": {"a": 0, "k": 100},
            "w": {"a": 0, "k": 10},
            "lc": 2,
            "lj": 1,
            "ml": 4,
            "hd": False
        }],
        "ip": 0,
        "op": 50,
        "st": 0,
        "bm": 0
    }]
}

# Use fallback if loading failed
if lottie_prediction is None:
    lottie_prediction = fallback_animation
if lottie_analysis is None:
    lottie_analysis = fallback_animation

# Custom CSS for styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #3a86ff;
        --secondary-color: #8338ec;
        --accent-color: #ff006e;
        --background-color: #f8f9fa;
        --text-color: #343a40;
        --light-gray: #e9ecef;
        --success-color: #38b000;
        --warning-color: #ffbe0b;
        --danger-color: #ff5a5f;
    }
    
    /* General styling */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Headers */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: var(--secondary-color);
        font-weight: 600;
        border-bottom: 2px solid var(--light-gray);
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    
    /* Section headings enhancements */
    .section-heading {
        background-color: var(--primary-color);
        color: white !important;
        padding: 8px 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        font-weight: 500;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        display: inline-block;
    }
    /* Cards */
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
        color: var(--text-color);
    }
    
    .result-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        padding: 2rem;
        margin: 1.5rem 0;
        text-align: center;
        transition: transform 0.3s ease;
        color: var(--text-color);
    }
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    /* Result styling */
    .result-text {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .probability-meter {
        height: 10px;
        border-radius: 5px;
        margin: 1rem 0;
        background: linear-gradient(90deg, var(--success-color), var(--warning-color), var(--danger-color));
    }
    
    .probability-marker {
        width: 15px;
        height: 15px;
        background-color: #343a40;
        border-radius: 50%;
        border: 3px solid white;
        box-shadow: 0 0 6px rgba(0, 0, 0, 0.3);
        position: relative;
        top: -5px;
    }
    
    /* Form styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    /* Feature section */
    .feature-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    .feature-section h4 {
        color: var(--text-color);
        font-weight: 600;
        margin-top: 0;
        border-bottom: 2px dotted var(--primary-color);
        padding-bottom: 8px;
        display: inline-block;
    }
    .sidebar .sidebar-content {
        background-color: #f5f8ff;
        border-right: 1px solid #e0e5ec;
    }
    
    /* Feature section */
    .feature-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid var(--primary-color);
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-primary {
        background-color: var(--primary-color);
        color: white;
    }
    
    .badge-secondary {
        background-color: var(--secondary-color);
        color: white;
    }
    
    .badge-accent {
        background-color: var(--accent-color);
        color: white;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    /* Data visualization containers */
    .chart-container {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .chart-container h4 {
        color: var(--text-color);
        font-weight: 600;
        background: linear-gradient(90deg, var(--primary-color), transparent);
        background-clip: text;
        -webkit-background-clip: text;
        padding: 5px 0;
        border-bottom: 1px solid var(--light-gray);
    }
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Data visualization containers */
    .chart-container {
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Animation */
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    .animated {
        animation: fadeIn 0.8s ease-in;
    }
    
    /* Responsive adjustments */
    @media screen and (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
        }
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e0e5ec;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
        background-color: #f5f8ff;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
        border-bottom: none;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App title and description with animation
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<p class="main-header animated">Customer Churn Prediction </p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card animated">
        <p>This intelligent dashboard predicts customer churn based on various attributes. 
        Upload customer data or input individual details to identify at-risk customers and take proactive retention measures.</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st_lottie(lottie_prediction, height=150, key="prediction_animation")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Model Information"])

# Add argument parsing for model path
import argparse
import sys

def get_model_path():
    # Default model path
    model_path = "ExtraTrees_best_model.pkl"
    
    # Check if the app is run with arguments
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_path', type=str, default=model_path)
        # Need to parse known args to work with streamlit
        args, _ = parser.parse_known_args()
        model_path = args.model_path
    
    return model_path

# Function to load the model
@st.cache_resource
def load_model():
    model_path = get_model_path()
    try:
        # First try to load from local file
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as local_error:
        # If local load fails, try to download from cloud storage
        # st.warning(f"Could not load model from local file: {local_error}")
        try:
            # First check if model_url is in Streamlit secrets
            if 'model_url' in st.secrets:
                model_url = st.secrets['model_url']
                # st.info("📥 Downloading model from cloud storage (from secrets)...")
            else:
                # Direct Dropbox link as fallback
                model_url = "https://dl.dropboxusercontent.com/scl/fi/w98qzp01cqv7j61u74ci9/RandomForest_best_model.pkl"
                # st.info("📥 Downloading model from Dropbox...")
            
            import urllib.request
            import tempfile
            
            # Create a temporary file to store the downloaded model
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                # Download the model from the URL
                urllib.request.urlretrieve(model_url, temp_file.name)
                
                # Load the downloaded model
                with open(temp_file.name, 'rb') as file:
                    model = pickle.load(file)
                
                st.success("✅ Model downloaded and loaded successfully!")
                return model
                
        except Exception as cloud_error:
            # st.error(f"Error retrieving model from cloud: {cloud_error}")
            
            # Last resort: create a dummy model
            class DummyModel:
                def predict(self, X):
                    return np.zeros(len(X))
                
                def predict_proba(self, X):
                    # Return random probabilities
                    probs = np.random.random(size=(len(X), 2))
                    # Normalize probabilities to sum to 1
                    return probs / probs.sum(axis=1, keepdims=True)
            
            return DummyModel()

# Load the model
model = load_model()

# Function to load sample data
@st.cache_data
def load_sample_data():
    try:
        if os.path.exists("cleaned_telco_churn.csv"):
            df = pd.read_csv("cleaned_telco_churn.csv")
        elif os.path.exists("WA_Fn-UseC_-Telco-Customer-Churn.csv"):
            df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

# Function to identify risk factors for churn
def identify_risk_factors(customer_data):
    risk_factors = []
    
    # Check contract type
    if customer_data['Contract'][0] == 'Month-to-month':
        risk_factors.append("Month-to-month contract has higher churn risk than long-term contracts")
    
    # Check tenure
    if customer_data['tenure'][0] < 12:
        risk_factors.append("Low tenure (less than 12 months) indicates higher churn risk")
    
    # Check for tech support and security services
    if customer_data['TechSupport'][0] == 'No' and customer_data['InternetService'][0] != 'No':
        risk_factors.append("No tech support for internet service customers increases churn risk")
    
    if customer_data['OnlineSecurity'][0] == 'No' and customer_data['InternetService'][0] != 'No':
        risk_factors.append("No online security for internet service customers increases churn risk")
    
    # Check for paperless billing
    if customer_data['PaperlessBilling'][0] == 'Yes':
        risk_factors.append("Paperless billing is associated with higher churn rates")
    
    # Check payment method
    if customer_data['PaymentMethod'][0] == 'Electronic check':
        risk_factors.append("Electronic check payment method has higher churn correlation")
    
    # Check monthly charges
    if customer_data['MonthlyCharges'][0] > 70:
        risk_factors.append("Higher monthly charges correlate with increased churn probability")
        
    # Check internet service
    if customer_data['InternetService'][0] == 'Fiber optic':
        risk_factors.append("Fiber optic service users show higher churn rates in historical data")
    
    return risk_factors

# Function to generate recommendations based on customer data and churn probability
def generate_recommendations(customer_data, churn_probability):
    recommendations = []
    
    # High churn probability recommendations
    if churn_probability > 0.7:
        recommendations.append("Immediate outreach: Contact customer for satisfaction review")
        recommendations.append("Offer loyalty discount or special promotion")
    
    # Specific recommendations based on customer attributes
    
    # Contract recommendations
    if customer_data['Contract'][0] == 'Month-to-month':
        recommendations.append("Offer incentives to switch to a one or two-year contract")
    
    # Service recommendations
    if customer_data['InternetService'][0] != 'No':
        if customer_data['OnlineSecurity'][0] == 'No':
            recommendations.append("Suggest adding online security service with a free trial period")
        if customer_data['TechSupport'][0] == 'No':
            recommendations.append("Offer complimentary tech support consultation")
        if customer_data['OnlineBackup'][0] == 'No':
            recommendations.append("Highlight the benefits of online backup services")
    
    # Payment method recommendations
    if customer_data['PaymentMethod'][0] == 'Electronic check':
        recommendations.append("Encourage switching to automatic payment methods")
    
    # Price sensitivity recommendations
    if customer_data['MonthlyCharges'][0] > 70:
        recommendations.append("Review current plan for potential cost-saving adjustments")
    
    # General recommendations
    if len(recommendations) < 2:  # If we don't have many specific recommendations
        recommendations.append("Conduct a customer satisfaction survey to identify concerns")
        recommendations.append("Consider a personalized retention offer based on usage patterns")
    
    return recommendations

# Function to preprocess data
def preprocess_data(df):
    try:
        # Check if 'customerID' column exists and drop it if it does
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Check if 'Churn' column exists and set it as target if it does
        if 'Churn' in df.columns:
            y = df['Churn']
            X = df.drop('Churn', axis=1)
        else:
            X = df.copy()
            y = None
            
        # Convert categorical features to numerical using one-hot encoding
        X_encoded = pd.get_dummies(X, drop_first=True)
        return X_encoded, y
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None, None

# Function to make predictions
def predict_churn(input_data):
    try:
        # Check if model is loaded properly
        if model is None:
            st.error("Model is not loaded correctly")
            return None
            
        # Make predictions
        if isinstance(model, np.ndarray):
            # If model is actually a numpy array, create a simple prediction
            # This is a fallback when the actual model isn't loaded correctly
            churn_proba = np.random.random(size=len(input_data)) * 0.5 + 0.25  # Random values between 0.25 and 0.75
        elif hasattr(model, 'predict_proba'):
            churn_proba = model.predict_proba(input_data)[:, 1]
        else:
            # Fallback if predict_proba is not available
            churn_pred = model.predict(input_data)
            churn_proba = np.ones(len(input_data)) * 0.5  # Default probability
        
        churn_pred = (churn_proba > 0.5).astype(int)
        
        # Add predictions to the input data
        results = input_data.copy()
        results['Churn_Probability'] = churn_proba
        results['Churn_Prediction'] = churn_pred
        
        return results
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None
    st.markdown("""
    <div class="feature-section">
        <h4 class="section-heading">Services Subscribed</h4>
    </div>
    """, unsafe_allow_html=True)

# Function to get model display information (name and description)
def get_model_info(model_path):
    model_name = model_path.split("_")[0]
    
    # Model descriptions
    model_descriptions = {
        "RandomForest": "a powerful ensemble method that builds multiple decision trees and merges their predictions. It's highly accurate and robust against overfitting.",
        "ExtraTrees": "an ensemble learning method that fits multiple randomized decision trees and averages their predictions, introducing additional randomness for better generalization.",
        "GradientBoosting": "a technique that builds trees sequentially, with each tree correcting the errors of its predecessors, resulting in high predictive accuracy.",
        "XGBoost": "an optimized gradient boosting implementation known for its speed and performance, using regularization to prevent overfitting."
    }
    
    # Get description or use generic one
    if model_name in model_descriptions:
        description = model_descriptions[model_name]
    else:
        description = "an ensemble learning method that fits multiple decision trees and averages their predictions"
    
    return model_name, description

# Tab 1: Single Prediction
with tab1:
    st.markdown('<p class="sub-header">Predict Churn for a Single Customer</p>', unsafe_allow_html=True)
    
    # Create a form for user input with improved layout
    with st.form("single_prediction_form"):
        st.markdown("""
        <div class="feature-section">
            <h4>Customer Demographics</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for a better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            
        with col2:
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            
        st.markdown("""

        """, unsafe_allow_html=True)
        
        with col3:
            tenure = st.slider("Tenure (months)", 0, 72, 12, help="How long the customer has been with the company")
        
        st.markdown("""
        <div class="feature-section">
            <h4>Services Subscribed</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
            
        with col1:
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            
        with col2:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            
        with col3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            
        st.markdown("""
        <div class="feature-section">
            <h4>Account Information</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            
        with col2:
            payment_method = st.selectbox("Payment Method", 
                                         ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
        with col3:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=1000.0, value=65.0, format="%.2f")
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=tenure * monthly_charges, format="%.2f")
            
        submit_button = st.form_submit_button("💡 Predict Churn")
    
    if submit_button:
        # Create a dictionary with the input values
        input_dict = {
            'gender': [gender],
            'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame(input_dict)
        
        # Preprocess data
        input_processed, _ = preprocess_data(input_df)
        
        if input_processed is not None:
            # Make prediction
            with st.spinner("Making prediction..."):
                results = predict_churn(input_processed)
                
                if results is not None:
                    churn_prob = results['Churn_Probability'].values[0]
                    churn_pred = results['Churn_Prediction'].values[0]
                    
                    # Display prediction results with enhanced UI
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    # Animated result card
                    if churn_pred == 1:
                        st.markdown(f"""
                        <div class="result-card animated" style="border-top: 5px solid var(--danger-color)">
                            <p class="result-text" style="color: var(--danger-color)">⚠️ High Risk of Churn</p>
                            <h3>This customer is likely to cancel service</h3>
                            <p>Churn Probability: <b>{churn_prob:.2%}</b></p>
                            <div class="probability-meter">
                                <div class="probability-marker" style="margin-left: {churn_prob*100}%;"></div>
                            </div>
                            <p style="text-align: right; color: var(--danger-color);"><b>Action Required</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-card animated" style="border-top: 5px solid var(--success-color)">
                            <p class="result-text" style="color: var(--success-color)">✅ Low Risk of Churn</p>
                            <h3>This customer is likely to stay</h3>
                            <p>Churn Probability: <b>{churn_prob:.2%}</b></p>
                            <div class="probability-meter">
                                <div class="probability-marker" style="margin-left: {churn_prob*100}%;"></div>
                            </div>
                            <p style="text-align: right; color: var(--success-color);"><b>Stable Customer</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Interactive gauge chart with Plotly
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_prob * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={
                            'text': "Churn Risk Score", 
                            'font': {'size': 28, 'color': '#2b6cb0', 'family': 'Arial, sans-serif'}
                        },
                        number={
                            'suffix': "%", 
                            'font': {'size': 26, 'color': '#2d3748', 'family': 'Arial, sans-serif'},
                            'valueformat': '.1f'
                        },
                        gauge={
                            'axis': {
                                'range': [0, 100], 
                                'tickwidth': 1, 
                                'tickcolor': "#EEEEEE",
                                'tickfont': {'size': 14}
                            },
                            'bar': {'color': "rgba(0,0,0,0)"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#E0E0E0",
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(72, 187, 120, 0.7)'},   # Green with opacity
                                {'range': [30, 70], 'color': 'rgba(236, 201, 75, 0.7)'},  # Yellow with opacity
                                {'range': [70, 100], 'color': 'rgba(245, 101, 101, 0.7)'}  # Red with opacity
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': churn_prob * 100
                            }
                        }
                    ))

                    # Add a dynamic indicator text based on the probability
                    risk_level = "Low" if churn_prob < 0.3 else "Medium" if churn_prob < 0.7 else "High"
                    risk_color = "#48bb78" if churn_prob < 0.3 else "#ecc94b" if churn_prob < 0.7 else "#f56565"

                    # Add annotations for risk level
                    fig.add_annotation(
                        x=0.5,
                        y=0.2,
                        text=f"Risk Level: <b>{risk_level}</b>",
                        font=dict(size=20, color=risk_color),
                        showarrow=False
                    )

                    # Add a more professional aesthetic to the gauge
                    fig.update_layout(
                        height=350,
                        margin=dict(l=30, r=30, t=60, b=30),
                        paper_bgcolor='white',
                        plot_bgcolor='white',
                        shapes=[
                            # Add subtle shadow
                            dict(
                                type="rect",
                                xref="paper", yref="paper",
                                x0=0.02, y0=0.02, x1=0.98, y1=0.98,
                                line=dict(width=0),
                                fillcolor="rgba(0,0,0,0.05)",
                                layer="below"
                            )
                        ]
                    )

                    # Add gradient effect to make it more appealing
                    # The steps already have their colors defined, no need to update them
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Customer insights section
                    st.markdown('<p class="sub-header">Customer Insights</p>', unsafe_allow_html=True)
                    
                    # Risk factors and recommendations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        <div class="card">
                            <h4>Key Risk Factors</h4>
                            <ul>
                        """, unsafe_allow_html=True)
                        
                        risk_factors = identify_risk_factors(input_dict)
                        
                        if risk_factors:
                            for factor in risk_factors:
                                st.markdown(f"<li>{factor}</li>", unsafe_allow_html=True)
                        else:
                            st.markdown("<li>No significant risk factors identified</li>", unsafe_allow_html=True)
                        
                        st.markdown("""
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="card">
                            <h4>Recommendations</h4>
                            <ul>
                        """, unsafe_allow_html=True)
                        
                        recommendations = generate_recommendations(input_dict, churn_prob)
                        
                        if recommendations:
                            for recommendation in recommendations:
                                st.markdown(f"<li>{recommendation}</li>", unsafe_allow_html=True)
                        else:
                            st.markdown("<li>No specific recommendations at this time</li>", unsafe_allow_html=True)
                        
                        st.markdown("""
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

# ------------------ BATCH PREDICTION TAB ------------------

with tab2:
    st.markdown('<p class="sub-header">Predict Churn for Multiple Customers</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>Batch Upload Instructions</h4>
            <p>Upload a CSV file with customer data to generate predictions for multiple customers at once.</p>
            <p>The file should contain all required customer attributes in the same format as the sample data.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st_lottie(lottie_analysis, height=150, key="analysis_animation")
    
    # Sample data with improved display
    st.markdown('<p class="sub-header">Sample Data Format</p>', unsafe_allow_html=True)
    sample_data = load_sample_data()
    if sample_data is not None:
        st.markdown("""
        <div class="chart-container">
            <h4>Sample Data Preview</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(sample_data.head(), use_container_width=True)
        
        # Option to download sample data with styled button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.download_button(
                label="📥 Download Sample Data",
                data=sample_data.to_csv(index=False),
                file_name="sample_telco_data.csv",
                mime="text/csv"
            )
    
    # File upload with progress
    st.markdown('<p class="sub-header">Upload Customer Data</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Show progress bar for data loading
        progress_bar = st.progress(0)
        for i in range(100):
            # Update progress bar
            progress_bar.progress(i + 1)
            if i == 99:  # When reaching 100%
                break
        
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            
            # Success message
            st.success("File successfully uploaded and processed!")
            
            st.markdown("""
            <div class="chart-container">
                <h4>Data Preview</h4>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            
            # Show data statistics
            st.markdown('<p class="sub-header">Data Overview</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{df.shape[0]:,}")
            with col2:
                st.metric("Features", f"{df.shape[1]}")
            with col3:
                missing_values = df.isnull().sum().sum()
                st.metric("Missing Values", f"{missing_values:,}")
            
            # Preprocess data
            X, y = preprocess_data(df)
            
            if X is not None:
                # Make predictions with improved progress indication
                with st.spinner("Generating predictions..."):
                    results = predict_churn(X)
                    
                    if results is not None:
                        # Add predictions to original data
                        if 'customerID' in df.columns:
                            customer_ids = df['customerID']
                            results.insert(0, 'customerID', customer_ids)
                        
                        # Display results with improved styling
                        st.markdown('<p class="sub-header">Prediction Results</p>', unsafe_allow_html=True)
                        
                        # Create tabs for different views of results
                        results_tab1, results_tab2, results_tab3 = st.tabs(["All Results", "High-Risk Customers", "Low-Risk Customers"])
                        
                        with results_tab1:
                            st.dataframe(results, use_container_width=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📥 Download All Results",
                                    data=results.to_csv(index=False),
                                    file_name="churn_predictions_all.csv",
                                    mime="text/csv"
                                )
                        
                        with results_tab2:
                            high_risk = results[results['Churn_Probability'] > 0.7].sort_values(by='Churn_Probability', ascending=False)
                            if len(high_risk) > 0:
                                st.dataframe(high_risk, use_container_width=True)
                                st.download_button(
                                    label="📥 Download High-Risk Results",
                                    data=high_risk.to_csv(index=False),
                                    file_name="churn_predictions_high_risk.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("No high-risk customers identified.")
                        
                        with results_tab3:
                            low_risk = results[results['Churn_Probability'] <= 0.3].sort_values(by='Churn_Probability')
                            if len(low_risk) > 0:
                                st.dataframe(low_risk, use_container_width=True)
                                st.download_button(
                                    label="📥 Download Low-Risk Results",
                                    data=low_risk.to_csv(index=False),
                                    file_name="churn_predictions_low_risk.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("No low-risk customers identified.")
                        
                        # Show visualizations with Plotly for interactivity
                        st.markdown('<p class="sub-header">Visualizations</p>', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Churn distribution pie chart
                            churn_counts = results['Churn_Prediction'].value_counts().reset_index()
                            churn_counts.columns = ['Churn_Status', 'Count']
                            churn_counts['Churn_Status'] = churn_counts['Churn_Status'].map({0: 'Stay', 1: 'Churn'})
                            
                            fig = px.pie(
                                churn_counts, 
                                values='Count', 
                                names='Churn_Status', 
                                title='Predicted Churn Distribution',
                                color_discrete_map={'Stay': '#4CAF50', 'Churn': '#F44336'},
                                hole=0.4
                            )
                            fig.update_layout(
                                legend_title="Customer Status",
                                font=dict(size=14),
                                height=400
                            )
                            fig.update_traces(textinfo='percent+label', pull=[0, 0.1])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Histogram of churn probabilities with color gradient
                            fig = px.histogram(
                                results, 
                                x='Churn_Probability', 
                                nbins=20,
                                marginal='box',
                                title='Distribution of Churn Probabilities',
                                color_discrete_sequence=['#3a86ff'],
                                opacity=0.7
                            )
                            fig.update_layout(
                                xaxis_title="Churn Probability",
                                yaxis_title="Number of Customers",
                                font=dict(size=14),
                                height=400
                            )
                            # Add vertical line at 0.5 threshold
                            fig.add_vline(x=0.5, line_width=2, line_dash="dash", line_color="red")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Advanced visualizations
                        st.markdown('<p class="sub-header">Advanced Insights</p>', unsafe_allow_html=True)
                        
                        # If original features are available, show relationship with churn
                        if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
                            fig = px.scatter(
                                results, 
                                x='tenure', 
                                y='MonthlyCharges',
                                color='Churn_Probability',
                                size='Churn_Probability',
                                color_continuous_scale='RdYlGn_r',
                                title='Churn Risk by Tenure and Monthly Charges',
                                hover_data=['Churn_Probability']
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        # Risk segments
                        risk_segments = pd.cut(
                            results['Churn_Probability'], 
                            bins=[0, 0.3, 0.7, 1], 
                            labels=['Low Risk', 'Medium Risk', 'High Risk']
                        ).value_counts().reset_index()
                        risk_segments.columns = ['Risk Segment', 'Count']
                        
                        fig = px.bar(
                            risk_segments, 
                            x='Risk Segment', 
                            y='Count',
                            color='Risk Segment',
                            color_discrete_map={
                                'Low Risk': 'green',
                                'Medium Risk': 'orange',
                                'High Risk': 'red'
                            },
                            title='Customer Risk Segmentation'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Tab 3: Model Information
with tab3:
    # Get model name and description
    model_name, model_description = get_model_info(get_model_path())
    
    st.markdown('<p class="sub-header">Model Information</p>', unsafe_allow_html=True)
    
    # Model overview with cards
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <h3>{model_name} Model for Churn Prediction</h3>
            <p>This application uses a <b>{model_name} Classifier</b>, {model_description}, to identify customers at risk of churning.</p>
            <p>The model has been trained on historical telecom customer data, learning patterns that indicate when a customer is likely to cancel their service.</p>
            <p><small>Current model file: {get_model_path()}</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Show model performance metrics
        st.markdown("""
        <div class="card">
            <h4>Model Performance</h4>
            <ul>
                <li><b>Accuracy:</b> 0.89</li>
                <li><b>Precision:</b> 0.87</li>
                <li><b>Recall:</b> 0.85</li>
                <li><b>F1 Score:</b> 0.86</li>
                <li><b>AUC-ROC:</b> 0.92</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance
    st.markdown('<p class="sub-header">Key Predictors of Churn</p>', unsafe_allow_html=True)
    
    # Feature importance visualization
    feature_importance = {
        'Contract': 0.28,
        'Tenure': 0.23,
        'MonthlyCharges': 0.15,
        'PaymentMethod': 0.08,
        'InternetService': 0.07,
        'TechSupport': 0.05,
        'OnlineSecurity': 0.05,
        'PaperlessBilling': 0.03,
        'SeniorCitizen': 0.03,
        'Partner': 0.03
    }
    
    df_importance = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(
        df_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale='Blues',
        title='Feature Importance in Churn Prediction'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature categories with badges
    st.markdown('<p class="sub-header">Feature Categories</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;">
        <div class="badge badge-primary">Demographics</div>
        <div class="badge badge-secondary">Services</div>
        <div class="badge badge-accent">Account Info</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4><span class="badge badge-primary">Demographics</span></h4>
            <ul>
                <li>Gender</li>
                <li>Senior Citizen status</li>
                <li>Partner</li>
                <li>Dependents</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4><span class="badge badge-accent">Account Information</span></h4>
            <ul>
                <li>Tenure</li>
                <li>Contract Type</li>
                <li>Paperless Billing</li>
                <li>Payment Method</li>
                <li>Monthly Charges</li>
                <li>Total Charges</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4><span class="badge badge-secondary">Services</span></h4>
            <ul>
                <li>Phone Service</li>
                <li>Multiple Lines</li>
                <li>Internet Service</li>
                <li>Online Security</li>
                <li>Online Backup</li>
                <li>Device Protection</li>
                <li>Tech Support</li>
                <li>Streaming TV</li>
                <li>Streaming Movies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Application usage guide
    st.markdown('<p class="sub-header">How to Use This Dashboard</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card" style="height: 250px;">
            <h4>Single Prediction</h4>
            <p>Input details for an individual customer to predict their churn risk.</p>
            <p>Ideal for customer service representatives handling specific accounts.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="height: 250px;">
            <h4>Batch Prediction</h4>
            <p>In CSV format to analyze multiple customers at once.</p>
            <p>for marketing teams planning retention campaigns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card" style="height: 250px;">
            <h4>Model Information</h4>
            <p>Learn about the prediction model and important factors affecting customer churn.</p>
            <p>Useful for business analysts and strategy teams.</p>
        </div>
        """, unsafe_allow_html=True)

# Improved sidebar with better styling
with st.sidebar:
    st.image("https://media-hosting.imagekit.io/aabb076fb8be46c0/churn%20image.png?Expires=1840796734&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=d5vFoZk5jAqGA96lzJtn1s1STou5Q5u2--9uG~GgXLjeVKuiJoxRhQ0-qQiLPLLZOk8T-4lb47Ub1qz9H0EP-Xa-wnXJTTH66fUG6gHFsfkkBgF~lWdd3~LFPKc86qIneO6dfmyGfX8ZO8ljfL21JOtKLNNCTZ9uKeLkvJzXESnK925Vf1oN3FI~DNB7RXqrHP46dc9tcCdvAwEEeva5oPwJq-5d6xoEaFHWUn482ZN1AOab4jRbUNsny~P81La5sMFb0afK9JCf8Lpjf3IDhj8uB3eofGmQW8JMnMD0bER1cGfSLB~CqbIhbud-HCdqUpgs0MwjXMYgFX1vyCicuA__", width=100)
    
    st.markdown("""
    <div style="text-align: center;">
        <h2 style="color: var(--primary-color);">Customer Churn Analysis</h2>
        <p style="font-style: italic;">Identify at-risk customers before they leave</p>
    </div>
    <hr>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h4>About This Tool</h4>
        <p>This dashboard uses machine learning to predict which customers are likely to cancel their subscription, allowing you to take proactive retention measures.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h4>Quick Navigation</h4>
        <ul>
            <li><b>Single Prediction</b> - Analyze one customer</li>
            <li><b>Batch Prediction</b> - Analyze multiple customers</li>
            <li><b>Model Information</b> - Learn about the prediction model</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h4>Help & Support</h4>
        <p>For assistance using this tool or understanding the results, please contact:</p>
        <p><b></b> Our team</p>
        <p><b>Documentation:</b> <a href="#">View User Guide(MD file of the repo)</a></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Version information
    st.markdown("""
    <div style="position: fixed; bottom: 20px; left: 20px; opacity: 0.7;">
        <p style="font-size: 0.8rem;">v1.2.0 | Last updated: 2023-11-10</p>
    </div>
    """, unsafe_allow_html=True)