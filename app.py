import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set page config
st.set_page_config(
    page_title="Depression Risk Assessment",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for premium aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Force text color to be dark since we have a light background */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #2c3e50 !important;
    }

    /* Titles */
    h1 {
        text-align: center;
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 2.5rem !important;
    }
    
    .subtitle {
        text-align: center;
        color: #5d6d7e !important;
        margin-bottom: 2rem;
        font-size: 1.2rem !important;
    }

    /* Form Container */
    [data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }

    /* Widgets */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        font-weight: 600;
        color: #34495e !important;
        font-size: 1rem !important;
    }
    
    /* Input fields background */
    .stSelectbox div[data-baseweb="select"] > div, .stNumberInput input {
        background-color: #f8f9fa !important;
        color: #2c3e50 !important;
        border: 1px solid #e0e0e0;
    }

    /* Fix Dropdown Menu Colors */
    div[data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    div[data-baseweb="popover"] div {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }
    
    div[data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    
    ul[data-baseweb="menu"] li {
        color: #2c3e50 !important;
        background-color: #ffffff !important;
    }
    
    ul[data-baseweb="menu"] li:hover, ul[data-baseweb="menu"] li[aria-selected="true"] {
        background-color: #f0f2f6 !important;
        color: #2c3e50 !important;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        color: white !important;
    }
    
    .stButton > button p {
        color: white !important;
    }

    /* Result Cards */
    .success-card {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        padding: 20px;
        border-radius: 15px;
        color: #1b5e20 !important;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .success-card h2, .success-card p {
        color: #1b5e20 !important;
    }
    
    .danger-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 99%, #fecfef 100%);
        padding: 20px;
        border-radius: 15px;
        color: #b71c1c !important;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .danger-card h2, .danger-card p {
        color: #b71c1c !important;
    }
    
    /* Warning message */
    .stAlert {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
""", unsafe_allow_html=True)

# Load model and encoders
@st.cache_resource
def load_artifacts():
    try:
        with open('depression_model.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        return artifacts
    except FileNotFoundError:
        return None

artifacts = load_artifacts()

if artifacts is None:
    st.error("⚠️ Model file 'depression_model.pkl' not found. Please run 'train_model.py' first.")
    st.stop()

model = artifacts['model']
encoders = artifacts['encoders']
feature_names = artifacts['features']

st.markdown("<h1>🎓 Student Wellness Monitor 🧠</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>🛡️ AI-Powered Depression Risk Assessment Tool for Students</p>", unsafe_allow_html=True)

# Create form for user input
with st.form("prediction_form"):
    st.markdown("### ✍️ Enter Student Details")
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        gender = st.selectbox("👤 Gender", encoders['Gender'].classes_)
        age = st.number_input("🎂 Age (Years)", min_value=15, max_value=60, value=20)
        city = st.selectbox("🏙️ City", encoders['City'].classes_)
        profession = st.selectbox("💼 Profession", encoders['Profession'].classes_)
        degree = st.selectbox("🎓 Degree", encoders['Degree'].classes_)
        academic_pressure = st.slider("📚 Academic Pressure (1-5)", 0.0, 5.0, 3.0)
        work_pressure = st.slider("💼 Work Pressure (0-5)", 0.0, 5.0, 0.0)
        cgpa = st.number_input("📈 CGPA (0-10)", 0.0, 10.0, 7.0)

    with col2:
        study_satisfaction = st.slider("😊 Study Satisfaction (1-5)", 0.0, 5.0, 3.0)
        job_satisfaction = st.slider("😌 Job Satisfaction (0-5)", 0.0, 5.0, 0.0)
        sleep_duration = st.selectbox("😴 Sleep Duration", encoders['Sleep Duration'].classes_)
        dietary_habits = st.selectbox("🍎 Dietary Habits", encoders['Dietary Habits'].classes_)
        suicidal_thoughts = st.selectbox("💭 History of Suicidal Thoughts?", encoders['Have you ever had suicidal thoughts ?'].classes_)
        work_study_hours = st.number_input("⏰ Daily Work/Study Hours", 0.0, 12.0, 6.0)
        financial_stress = st.slider("💸 Financial Stress (1-5)", 0.0, 5.0, 3.0)
        family_history = st.selectbox("👪 Family History of Mental Illness", encoders['Family History of Mental Illness'].classes_)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🚀 Analyze Risk Profile")

if submitted:
    # Prepare input data
    input_data = {
        'Gender': gender,
        'Age': age,
        'City': city,
        'Profession': profession,
        'Academic Pressure': academic_pressure,
        'Work Pressure': work_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_satisfaction,
        'Job Satisfaction': job_satisfaction,
        'Sleep Duration': sleep_duration,
        'Dietary Habits': dietary_habits,
        'Degree': degree,
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Work/Study Hours': work_study_hours,
        'Financial Stress': financial_stress,
        'Family History of Mental Illness': family_history
    }
    
    # Encode categorical variables
    input_df = pd.DataFrame([input_data])
    
    try:
        for col, le in encoders.items():
            if col in input_df.columns:
                 input_df[col] = le.transform(input_df[col])
        
        # Ensure correct column order
        input_df = input_df[feature_names]
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        st.markdown("---")
        
        if prediction == 1:
            st.markdown(f"""
                <div class="danger-card">
                    <h2>⚠️ High Risk Detected</h2>
                    <p>The model predicts a high likelihood of depression (Probability: {probability:.2%}).</p>
                    <p><strong>Recommendation:</strong> successful intervention often involves consulting a mental health professional.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="success-card">
                    <h2>✅ Low Risk Detected</h2>
                    <p>The model predicts a low likelihood of depression (Probability: {probability:.2%}).</p>
                    <p>Maintaining a healthy work-life balance is key to mental wellness.</p>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
