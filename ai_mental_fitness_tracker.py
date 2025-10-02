# MindBalance - Wellbeing Check-in (Final Corrected Logic)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

# (Data loading and model training code remains for context, but is NOT used for final score)

# -----------------------
# Data load & preparation
# -----------------------
DF_PREV = "prevalence-by-mental-and-substance-use-disorder.csv"
DF_SHARE = "mental-and-substance-use-as-share-of-disease.csv"

try:
    df1 = pd.read_csv(DF_PREV)
    df2 = pd.read_csv(DF_SHARE)
except FileNotFoundError:
    st.error(
        f"Error: Please make sure '{DF_PREV}' and '{DF_SHARE}' are in the same directory.")
    st.stop()

data = pd.merge(df1, df2)
if "Code" in data.columns:
    data = data.drop(columns=["Code"])
data = data.set_axis([
    "Country", "Year", "Schizophrenia", "Bipolar-disorder",
    "Eating-disorders", "Anxiety-disorders", "Drug-use disorders",
    "Depressive-disorders", "Alcohol-use disorders", "Mental-Fitness"
], axis='columns')

# -----------------------
# Helper Functions
# -----------------------
OPTIONS = ["No", "Not Really", "Don't Know", "Sure", "Yes"]


def safe_map(ans):
    """
    Maps user's answer to a 0-100 risk scale.
    "No" and "Don't Know" are 0 risk.
    "Not Really" is a low risk (25).
    "Sure" and "Yes" indicate higher risk (75 and 100).
    """
    if ans == "No" or ans == "Don't Know":
        return 0.0
    elif ans == "Not Really":
        return 25.0
    elif ans == "Sure":
        return 75.0
    elif ans == "Yes":
        return 100.0
    return 0.0


def get_targeted_tips(vals):
    tips = []
    if vals["Anxiety-disorders"] >= 75:
        tips.append(
            "Pause for a 2-minute breathing exercise whenever stress rises.")
    if vals["Depressive-disorders"] >= 75:
        tips.append(
            "Start with small daily goals—simple routines can lift your mood.")
    if vals["Alcohol-use disorders"] >= 75 or vals["Drug-use disorders"] >= 75:
        tips.append(
            "It may help to talk with a support group or professional about substance use.")
    if vals["Eating-disorders"] >= 75:
        tips.append(
            "If eating habits feel unbalanced, a nutritionist or counselor can guide you.")
    if vals["Schizophrenia"] >= 75 or vals["Bipolar-disorder"] >= 75:
        tips.append(
            "Strong mood changes or unusual perceptions deserve professional support.")

    if not tips:
        tips = [
            "Go for a short walk—movement clears the mind and resets focus.",
            "Try journaling: note one positive moment and one small challenge today.",
            "Aim for 7–8 hours of steady sleep to recharge your mind.",
            "Connect with a friend or family member, even briefly—it boosts mood."
        ]
    return tips[:4]


# -----------------------
# Streamlit Interface
# -----------------------
st.set_page_config(page_title="MindBalance", page_icon="🧠", layout="centered")
st.title("🧠 MindBalance - Your Wellbeing Check-in")
st.write("Take a short reflection to understand your mental wellbeing better. This is not a diagnosis - just gentle guidance. Be honest, this is only for you.")

with st.form("checkin_form"):
    st.subheader("Your Honest Reflection")

    # Country and Year inputs are kept for user context but not used in the new logic
    countries = sorted(data["Country"].unique())
    country = st.selectbox("Where do you live?", [
                           "Prefer not to say"] + list(countries), index=0)
    year = st.number_input(
        "Year", value=2025, min_value=1990, max_value=2100, step=1)

    schizo = st.radio("How often have you had unusual thoughts or perceptions?",
                      OPTIONS, horizontal=True, index=2)
    bipolar = st.radio("How often have your moods swung between extreme highs and lows?",
                       OPTIONS, horizontal=True, index=2)
    eating = st.radio("How often have your eating habits felt unhealthy lately?",
                      OPTIONS, horizontal=True, index=2)
    anxiety = st.radio("How often do you feel more anxious than usual?",
                       OPTIONS, horizontal=True, index=2)
    drug_use = st.radio(
        "How often have you used drugs or substances more than you'd like?", OPTIONS, horizontal=True, index=2)
    depression = st.radio(
        "How often do you feel low or lose interest in things?", OPTIONS, horizontal=True, index=2)
    alcohol = st.radio("How often have you been drinking alcohol more than usual?",
                       OPTIONS, horizontal=True, index=2)

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

    # --- SCORING LOGIC - FIXED ---
    # The model is no longer used. Score is based purely on the average of answers.
    risk_score = np.mean(list(input_vals.values()))
    final_score = 100.0 - risk_score
    # --- END OF FIX ---

    st.subheader("📊 Your Wellbeing Snapshot")

    if final_score < 40:
        st.error("⚠️ Your wellbeing seems **at risk**. Please take extra care.")
        header_msg = "At Risk"
    elif final_score < 70:
        st.warning(
            "🙂 You’re **doing okay**, but there’s room to strengthen your wellbeing.")
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
        st.write(
            "> If you feel unsafe, please reach out to emergency services or a trusted crisis line immediately.")
        st.write(
            "> Talking with a licensed professional could make a real difference.")
    else:
        st.write("> These are reflective suggestions, not medical advice. If you feel ongoing struggles, professional support is always valuable.")

    st.caption("💡 MindBalance is a reflection tool, not a medical diagnosis.")
