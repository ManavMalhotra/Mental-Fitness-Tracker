import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import time

# ----------------------------
# Load & prepare dataset
# ----------------------------
df2 = pd.read_csv("mental-and-substance-use-as-share-of-disease.csv")
df1 = pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")

data = pd.merge(df1, df2)
data.drop('Code', axis=1, inplace=True)

data = data.set_axis([
    "Country","Year","Schizophrenia","Bipolar-disorder",
    "Eating-disorders","Anxiety-disorders","Drug-use disorders",
    "Depressive-disorders","Alcohol-use disorders","Mental-Fitness"
], axis ='columns')

lab = LabelEncoder()
for i in data.columns:
    if data[i].dtype == 'object':
        data[i] = lab.fit_transform(data[i])

x = data.drop('Mental-Fitness',axis=1)
y = data['Mental-Fitness']

xtrain, xtest, ytrain, ytest = train_test_split(x, y , test_size=.20, random_state=2)

# ----------------------------
# Train models
# ----------------------------
lr = LinearRegression()
lr.fit(xtrain,ytrain)

rf = RandomForestRegressor()
rf.fit(xtrain,ytrain)

svr = SVR()
svr.fit(xtrain,ytrain)

# ----------------------------
# Streamlit App
# ----------------------------
st.title("🧠 Mental Fitness Tracker")
st.write("Answer a few quick questions to check your mental fitness score. No right or wrong answers — just be honest.")

Country = st.selectbox("🌍 Which country do you live in?", data["Country"].unique())
Year = 2025  # default year

# ----------------------------
# Human-like questions
# ----------------------------
schizo = st.radio("Do you ever feel disconnected from reality, like things around you aren’t real?",
                  ["Not at all", "Sometimes", "Often", "Not sure"])

bipolar = st.radio("Do your moods swing between extreme highs and deep lows?",
                   ["No", "Occasionally", "Yes, quite a lot", "Not sure"])

eating = st.radio("Have your eating habits changed — like eating too little or too much lately?",
                  ["No", "A little", "Yes, noticeable", "Not sure"])

anxiety = st.radio("Do you feel restless, anxious, or on edge most of the time?",
                   ["Not really", "Sometimes", "Often", "Not sure"])

drug_use = st.radio("How often do you use recreational substances (like drugs)?",
                    ["Never", "Rarely", "Frequently", "Not sure"])

depression = st.radio("Do you often feel sad, hopeless, or lose interest in things you usually enjoy?",
                      ["Not at all", "Sometimes", "Often", "Not sure"])

alcohol = st.radio("How often do you consume alcohol?",
                   ["Never", "Occasionally", "Frequently", "Not sure"])

# ----------------------------
# Mapping answers -> numeric values
# ----------------------------
answer_map = {
    "Not at all": 0, "No": 0, "Never": 0, "Not really": 10,
    "Rarely": 20, "A little": 30, "Sometimes": 50, "Occasionally": 50,
    "Often": 70, "Yes, noticeable": 70, "Yes, quite a lot": 80,
    "Frequently": 90, "Not sure": 40
}

schizo_val = answer_map[schizo]
bipolar_val = answer_map[bipolar]
eating_val = answer_map[eating]
anxiety_val = answer_map[anxiety]
drug_val = answer_map[drug_use]
depression_val = answer_map[depression]
alcohol_val = answer_map[alcohol]

# ----------------------------
# Prediction & Results
# ----------------------------
if st.button("🔍 Calculate Mental Fitness"):
    with st.spinner("Analyzing your responses..."):
        time.sleep(2)

    country_encoded = lab.transform([Country])[0]
    inputData = pd.DataFrame({
        'Country': [country_encoded],
        'Year': [Year],
        'Schizophrenia': [schizo_val],
        'Bipolar-disorder': [bipolar_val],
        'Eating-disorders': [eating_val],
        'Anxiety-disorders': [anxiety_val],
        'Drug-use disorders': [drug_val],
        'Depressive-disorders': [depression_val],
        'Alcohol-use disorders': [alcohol_val]
    })

    # Predictions
    pred_lr = lr.predict(inputData)[0] * 10
    pred_rf = rf.predict(inputData)[0] * 10
    pred_svm = svr.predict(inputData)[0] * 10

    st.subheader("📊 Your Mental Fitness Predictions")
    st.write(f"**Linear Regression:** {pred_lr:.2f}%")
    st.write(f"**Random Forest:** {pred_rf:.2f}%")
    st.write(f"**SVM Regression:** {pred_svm:.2f}%")

    # Final suggestion (based on RF since it's usually stronger)
    final_score = pred_rf
    st.subheader("💡 Suggestion for You")
    if final_score < 40:
        st.error("⚠️ Low score. Please consider professional help and focus on self-care routines.")
    elif final_score < 70:
        st.warning("🙂 Moderate score. Try improving sleep, reducing stress, and connecting with supportive people.")
    else:
        st.success("🎉 Great! You’re maintaining good mental health habits. Keep it up!")
