# MindBalance - Wellbeing Check-in
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

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

# Merge and clean
data = pd.merge(df1, df2)
if "Code" in data.columns:
    data = data.drop(columns=["Code"])

data = data.set_axis([
    "Country", "Year", "Schizophrenia", "Bipolar-disorder",
    "Eating-disorders", "Anxiety-disorders", "Drug-use disorders",
    "Depressive-disorders", "Alcohol-use disorders", "Mental-Fitness"
], axis='columns')

FEATURE_COLS = [
    "Schizophrenia", "Bipolar-disorder", "Eating-disorders",
    "Anxiety-disorders", "Drug-use disorders", "Depressive-disorders",
    "Alcohol-use disorders"
]
TARGET_COL = "Mental-Fitness"

data = data.dropna(subset=FEATURE_COLS + [TARGET_COL])
X = data[FEATURE_COLS].astype(float).copy()
y = data[TARGET_COL].astype(float).copy()

xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42)

# -----------------------
# Train the Model
# -----------------------
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)
model.fit(xtrain, ytrain)

train_preds = model.predict(xtrain)
model_min = float(np.min(train_preds))
model_max = float(np.max(train_preds))
if model_max == model_min:
    model_max += 1.0


def model_to_percent(raw_pred):
    pct = (raw_pred - model_min) / (model_max - model_min)
    return float(np.clip(pct * 100.0, 0.0, 100.0))


# -----------------------
# Helper Functions
# -----------------------
OPTIONS = ["No", "Not Really", "Don't Know", "Sure", "Yes"]


def safe_map(ans):
    if ans not in OPTIONS:
        return 0.0
    idx = OPTIONS.index(ans)
    return float(idx / (len(OPTIONS) - 1) * 100.0)


def get_targeted_tips(vals):
    tips = []
    if vals["Anxiety-disorders"] >= 60:
        tips.append(
            "Pause for a 2-minute breathing exercise whenever stress rises.")
    if vals["Depressive-disorders"] >= 60:
        tips.append(
            "Start with small daily goals-simple routines can lift your mood.")
    if vals["Alcohol-use disorders"] >= 60 or vals["Drug-use disorders"] >= 60:
        tips.append(
            "It may help to talk with a support group or professional about substance use.")
    if vals["Eating-disorders"] >= 60:
        tips.append(
            "If eating habits feel unbalanced, a nutritionist or counselor can guide you.")
    if vals["Schizophrenia"] >= 60 or vals["Bipolar-disorder"] >= 60:
        tips.append(
            "Strong mood changes or unusual perceptions deserve professional support.")

    if not tips:
        tips = [
            "Go for a short walk-movement clears the mind and resets focus.",
            "Try journaling: note one positive moment and one small challenge today.",
            "Aim for 7–8 hours of steady sleep to recharge your mind.",
            "Connect with a friend or family member, even briefly-it boosts mood."
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

    countries = sorted(data["Country"].unique())
    country = st.selectbox(
        "Where do you live?",
        ["Prefer not to say"] + list(countries),
        index=(["Prefer not to say"] + list(countries)).index("India"))

    year = st.number_input("Year", value=2025,
                           min_value=1990, max_value=2100, step=1)

    schizo = st.radio("Have you experienced unusual thoughts or perceptions?",
                      OPTIONS, horizontal=True, index=None)
    bipolar = st.radio("Do your moods swing between extreme highs and lows?",
                       OPTIONS, horizontal=True, index=None)
    eating = st.radio("Have your eating habits felt unhealthy lately?",
                      OPTIONS, horizontal=True, index=None)
    anxiety = st.radio("Do you often feel more anxious than usual?",
                       OPTIONS, horizontal=True, index=None)
    drug_use = st.radio("Have you been using drugs or substances more than you'd like?",
                        OPTIONS, horizontal=True, index=None)
    depression = st.radio(
        "Do you frequently feel low or lose interest in things?", OPTIONS, horizontal=True, index=None)
    alcohol = st.radio("Have you been drinking alcohol more than usual?",
                       OPTIONS, horizontal=True, index=None)

    submitted = st.form_submit_button("Analyze My Wellbeing")

if submitted:
    all_answers = [schizo, bipolar, eating,
                   anxiety, drug_use, depression, alcohol]
    if any(answer is None for answer in all_answers):
        st.error("⚠️ Please answer all questions before continuing.")
    else:
        input_vals = {
            "Schizophrenia": safe_map(schizo),
            "Bipolar-disorder": safe_map(bipolar),
            "Eating-disorders": safe_map(eating),
            "Anxiety-disorders": safe_map(anxiety),
            "Drug-use disorders": safe_map(drug_use),
            "Depressive-disorders": safe_map(depression),
            "Alcohol-use disorders": safe_map(alcohol)
        }

        with st.spinner("✨ Analyzing your responses... Gathering insights... Almost there..."):
            time.sleep(2.5)

        model_input = pd.DataFrame(
            [input_vals], columns=FEATURE_COLS).astype(float)
        model_pred_raw = model.predict(model_input)[0]
        model_risk_pct = model_to_percent(model_pred_raw)
        manual_risk_pct = np.mean(list(input_vals.values()))

        if manual_risk_pct >= 80:
            w_model, w_manual = 0.3, 0.7
        elif manual_risk_pct >= 50:
            w_model, w_manual = 0.4, 0.6
        else:
            w_model, w_manual = 0.6, 0.4

        risk_score = float(np.clip(w_model * model_risk_pct +
                           w_manual * manual_risk_pct, 0.0, 100.0))
        final_score = 100.0 - risk_score

        st.subheader("📊 Your Wellbeing Snapshot")

        if final_score < 40:
            st.error("⚠️ Your wellbeing seems **at risk**. Please take extra care.")
            header_msg = "At Risk"
        elif final_score < 70:
            st.warning(
                "🙂 You’re **doing okay**, but there’s room to strengthen your wellbeing.")
            header_msg = "Doing Okay"
        else:
            st.success(
                "🎉 Your wellbeing looks **strong and steady**. Keep it up!")
            header_msg = "Strong Wellbeing"

        st.markdown(
            f"**Status:** {header_msg} - **Score:** {final_score:.1f}/100")
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
