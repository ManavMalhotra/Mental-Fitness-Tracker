import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder



df2 = pd.read_csv("mental-and-substance-use-as-share-of-disease.csv")
df1 = pd.read_csv("prevalence-by-mental-and-substance-use-disorder.csv")

df1.head()

df2.head(10)

data = pd.merge(df1,df2)
data.head(10)


data.isnull().sum()

data.drop('Code', axis=1, inplace=True)
data.head(10)

data = data.set_axis(["Country","Year","Schizophrenia","Bipolar-disorder","Eating-disorders","Anxiety-disorders","Drug-use disorders","Depressive-disorders","Alcohol-use disorders","Mental-Fitness"], axis ='columns')
data.head(10)

from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()
for i in data.columns:
  if data[i].dtype == 'object':
    data[i] = lab.fit_transform(data[i])


x = data.drop('Mental-Fitness',axis=1)
y = data['Mental-Fitness']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y , test_size=.20, random_state=2)

print("xtrain =", xtrain.shape)
print("xtest =", xtest.shape)
print()
print("ytrain =", ytrain.shape)
print("ytest =", ytest.shape)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
lr = LinearRegression()
lr.fit(xtrain,ytrain)

ytrain_pred = lr.predict(xtrain)
mse = mean_squared_error(ytrain,ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain,ytrain_pred)))
r2 = r2_score(ytrain,ytrain_pred)

print("The Linear Regression model performance for training set")
print("-------------------------------------------------")
print("MSE is {}".format(mse))
print("RMSE is {}".format(rmse))
print("R2 score is {}".format(r2))


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xtrain,ytrain)

ytrain_pred = rf.predict(xtrain)
mse = mean_squared_error(ytrain,ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain,ytrain_pred)))
r2 = r2_score(ytrain,ytrain_pred)

print("The Random Forest Regressor model performance for training set")
print("-------------------------------------------------")
print("MSE is {}".format(mse))
print("RMSE is {}".format(rmse))
print("R2 score is {}".format(r2))


from sklearn.svm import SVR
svr=SVR()
svr.fit(xtrain,ytrain)

ytrain_pred = svr.predict(xtrain)
mse = mean_squared_error(ytrain,ytrain_pred)
rmse = (np.sqrt(mean_squared_error(ytrain,ytrain_pred)))
r2 = r2_score(ytrain,ytrain_pred)

print("The SVM Regression model performance for training set")
print("-------------------------------------------------")
print("MSE is {}".format(mse))
print("RMSE is {}".format(rmse))
print("R2 score is {}".format(r2))


ytest_pred = lr.predict(xtest)
mse = mean_squared_error(ytest,ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest,ytest_pred)))
r2 = r2_score(ytest,ytest_pred)

print("The Linear Regression model performance for training set")
print("-------------------------------------------------")
print("MSE is {}".format(mse))
print("RMSE is {}".format(rmse))
print("R2 score is {}".format(r2))

ytest_pred = rf.predict(xtest)
mse = mean_squared_error(ytest,ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest,ytest_pred)))
r2 = r2_score(ytest,ytest_pred)

print()

print("The Random Forest Regressor model performance for training set")
print("-------------------------------------------------")
print("MSE is {}".format(mse))
print("RMSE is {}".format(rmse))
print("R2 score is {}".format(r2))

ytest_pred = svr.predict(xtest)
mse = mean_squared_error(ytest,ytest_pred)
rmse = (np.sqrt(mean_squared_error(ytest,ytest_pred)))
r2 = r2_score(ytest,ytest_pred)

print()


print("The SVM Regression model performance for training set")
print("-------------------------------------------------")
print("MSE is {}".format(mse))
print("RMSE is {}".format(rmse))
print("R2 score is {}".format(r2))



import streamlit as st

st.write("------  Welcome to Mental Fitness Tracker ------")
st.write("------ Please Enter the Following Details ------ \n")

Country= st.text_input('Enter Your country Name: ')
Year= st.number_input("Enter the Year:",min_value=0.0, format="%.2f")
Schizophrenia = 78
Bipolar= 69
Eating= st.number_input("Eating disorder rate in %: ",min_value=0.0, format="%.2f")
Anxiety= st.number_input("Anxiety rate in %: ", min_value=0.0,format="%.2f")
Drug_use= st.number_input("Drug Usage rate in per year %: ",min_value=0.0, format="%.2f")
Depression= st.number_input("Depression rate in %: ", min_value=0.0,format="%.2f")
Alcohol= st.number_input("Alcohol Consuming rate per year in %: ",min_value=0.0, format="%.2f")




if st.button("Calculate"):
    Country = lab.fit_transform([Country])
    inputData = pd.DataFrame({'Country':Country,'Year':Year,'Schizophrenia':Schizophrenia,'Bipolar-disorder':Bipolar,'Eating disorders':Eating,'Anxiety-disorders':Anxiety,'Drug-use disorders':Drug_use,'Depressive-disorders':Depression,'Alcohol-use disorders':Alcohol},index=[0])
    prediction = rf.predict(inputData.values)

    st.write("Your Mental Fitness is {}%".format(prediction*10))



