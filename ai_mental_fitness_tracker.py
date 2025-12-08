import streamlit as st

# ⚠️ THIS MUST BE THE ABSOLUTE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="MindBal", page_icon="🧠", layout="centered")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Load and Prepare Data
# -----------------------
@st.cache_resource
def load_and_train():
    try:
        df1 = pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")
        df2 = pd.read_csv("mental-and-substance-use-as-share-of-disease.csv")
    except FileNotFoundError:
        return None, None, None, "CSV files not found"
    
    # Merge data
    data = pd.merge(df1, df2)
    
    # Remove Code column if exists
    if "Code" in data.columns:
        data = data.drop(columns=["Code"])
    
    # Rename columns
    data.columns = [
        "Country", "Year", "Schizophrenia", "Bipolar-disorder",
        "Eating-disorders", "Anxiety-disorders", "Drug-use disorders",
        "Depressive-disorders", "Alcohol-use disorders", "Mental-Fitness"
    ]
    
    # Encode countries
    le = LabelEncoder()
    data["Country_Encoded"] = le.fit_transform(data["Country"])
    
    # Prepare features and target
    X = data[["Country_Encoded", "Year", "Schizophrenia", "Bipolar-disorder",
              "Eating-disorders", "Anxiety-disorders", "Drug-use disorders",
              "Depressive-disorders", "Alcohol-use disorders"]]
    y = data["Mental-Fitness"]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    countries = sorted(data["Country"].unique().tolist())
    country_map = {country: le.transform([country])[0] for country in countries}
    
    return model, countries, country_map, None

# Load model and data
model, countries, country_to_code, error = load_and_train()

if error:
    st.error(f"Error: {error}")
    st.info("Please ensure 'prevalence-by-mental-and-substance-use-disorder.csv' and 'mental-and-substance-use-as-share-of-disease.csv' are in the app directory.")
    st.stop()

# -----------------------
# Constants
# -----------------------
OPTIONS = ["No", "Not Really", "Don't Know", "Sure", "Yes"]

PREVALENCE_RANGES = {
    "Schizophrenia": (0.20, 0.35),
    "Bipolar-disorder": (0.65, 0.95),
    "Eating-disorders": (0.08, 0.35),
    "Anxiety-disorders": (3.50, 7.50),
    "Drug-use disorders": (0.20, 1.80),
    "Depressive-disorders": (3.50, 7.00),
    "Alcohol-use disorders": (0.30, 2.50)
}

# -----------------------
# Helper Functions
# -----------------------
def map_to_prevalence(answer, disorder):
    min_val, max_val = PREVALENCE_RANGES[disorder]
    scale = {"No": 0.0, "Not Really": 0.25, "Don't Know": 0.5, "Sure": 0.75, "Yes": 1.0}
    factor = scale.get(answer, 0.5)
    return min_val + (max_val - min_val) * factor

def safe_map(answer):
    mapping = {"No": 0.0, "Not Really": 25.0, "Don't Know": 50.0, "Sure": 75.0, "Yes": 100.0}
    return mapping.get(answer, 50.0)

def get_targeted_tips(vals):
    tips = []
    if vals["Anxiety-disorders"] >= 75:
        tips.append("Pause for a 2-minute breathing exercise whenever stress rises.")
    if vals["Depressive-disorders"] >= 75:
        tips.append("Start with small daily goals—simple routines can lift your mood.")
    if vals["Alcohol-use disorders"] >= 75 or vals["Drug-use disorders"] >= 75:
        tips.append("It may help to talk with a support group or professional about substance use.")
    if vals["Eating-disorders"] >= 75:
        tips.append("If eating habits feel unbalanced, a nutritionist or counselor can guide you.")
    if vals["Schizophrenia"] >= 75 or vals["Bipolar-disorder"] >= 75:
        tips.append("Strong mood changes or unusual perceptions deserve professional support.")
    
    if not tips:
        tips = [
            "Go for a short walk—movement clears the mind and resets focus.",
            "Try journaling: note one positive moment and one small challenge today.",
            "Aim for 7–8 hours of steady sleep to recharge your mind.",
            "Connect with a friend or family member, even briefly—it boosts mood."
        ]
    return tips[:4]

def encode_country(country_name):
    if country_name == "Prefer not to say":
        return 0
    return country_to_code.get(country_name, 0)

# -----------------------
# Streamlit UI
# -----------------------
st.title("🧠 MindBal: Wellbeing Assessment System")
st.write("Take a short reflection to understand your mental wellbeing better. This is not a diagnosis - just gentle guidance. Be honest, this is only for you.")

with st.form("checkin_form"):
    st.subheader("Your Honest Reflection")
    
    country = st.selectbox("Where do you live?", ["Prefer not to say"] + countries, index=0)
    year = st.number_input("Year", value=2025, min_value=1990, max_value=2100, step=1)
    
    schizo = st.radio("How often have you had unusual thoughts or perceptions?", OPTIONS, horizontal=True, index=2)
    bipolar = st.radio("How often have your moods swung between extreme highs and lows?", OPTIONS, horizontal=True, index=2)
    eating = st.radio("How often have your eating habits felt unhealthy lately?", OPTIONS, horizontal=True, index=2)
    anxiety = st.radio("How often do you feel more anxious than usual?", OPTIONS, horizontal=True, index=2)
    drug_use = st.radio("How often have you used drugs or substances more than you'd like?", OPTIONS, horizontal=True, index=2)
    depression = st.radio("How often do you feel low or lose interest in things?", OPTIONS, horizontal=True, index=2)
    alcohol = st.radio("How often have you been drinking alcohol more than usual?", OPTIONS, horizontal=True, index=2)
    
    submitted = st.form_submit_button("Analyze My Wellbeing")

if submitted:
    input_vals = {
        "Schizophrenia": safe_map(schizo),
        "Bipolar-disorder": safe_map(bipolar),
        "Eating-disorders": safe_map(eating),
        "Anxiety-disorders": safe_map(anxiety),
        "Drug-use disorders": safe_map(drug_use),
        "Depressive-disorders": safe_map(depression),
        "Alcohol-use disorders": safe_map(alcohol)
    }
    
    with st.spinner("✨ Analyzing your responses..."):
        time.sleep(1.5)
    
    try:
        country_encoded = encode_country(country)
        
        features = [
            country_encoded,
            year,
            map_to_prevalence(schizo, "Schizophrenia"),
            map_to_prevalence(bipolar, "Bipolar-disorder"),
            map_to_prevalence(eating, "Eating-disorders"),
            map_to_prevalence(anxiety, "Anxiety-disorders"),
            map_to_prevalence(drug_use, "Drug-use disorders"),
            map_to_prevalence(depression, "Depressive-disorders"),
            map_to_prevalence(alcohol, "Alcohol-use disorders")
        ]
        
        prediction = model.predict([features])[0]
        raw_score = prediction * 10
        final_score = max(0, min(100, 100 - raw_score))
        
    except Exception:
        risk_score = np.mean(list(input_vals.values()))
        final_score = 100.0 - risk_score
    
    st.subheader("📊 Your Wellbeing Snapshot")
    
    if final_score < 40:
        st.error("⚠️ Your wellbeing seems **at risk**. Please take extra care.")
        header_msg = "At Risk"
    elif final_score < 70:
        st.warning("🙂 You're **doing okay**, but there's room to strengthen your wellbeing.")
        header_msg = "Doing Okay"
    else:
        st.success("🎉 Your wellbeing looks **strong and steady**. Keep it up!")
        header_msg = "Strong Wellbeing"
    
    st.markdown(f"**Status:** {header_msg} - **Score:** {final_score:.1f}/100")
    st.progress(int(final_score))
    
    tips = get_targeted_tips(input_vals)
    st.write("### 🌱 Personalized Recommendations")
    for t in tips:
        st.write(f"- {t}")
    
    if final_score < 40:
        st.write("> If you feel unsafe, please reach out to emergency services or a trusted crisis line immediately.")
        st.write("> Talking with a licensed professional could make a real difference.")
    else:
        st.write("> These are reflective suggestions, not medical advice. If you feel ongoing struggles, professional support is always valuable.")
    
    st.caption("💡 MindBal is a reflection tool, not a medical diagnosis.")