import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder


"""# Preparing and Loading the data"""

df1 = pd.read_csv("./prevalence-by-mental-and-substance-use-disorder.csv")
df2 = pd.read_csv("./mental-and-substance-use-as-share-of-disease.csv")

df1.head()

df2.head(10)

data = pd.merge(df1,df2)
data.head(10)

"""# Data Cleaning"""

data.isnull().sum()

data.drop('Code', axis=1, inplace=True)
data.head(10)

# data.size,data.shape

data = data.set_axis(["Country","Year","Schizophrenia","Bipolar-disorder","Eating-disorders","Anxiety-disorders","Drug-use disorders","Depressive-disorders","Alcohol-use disorders","Mental-Fitness"], axis ='columns')
data.head(10)

# """#Visualization"""

# plt.figure(figsize=(12,6))
# sb.heatmap(data.corr(),annot=True,cmap='Reds')
# plt.plot()

# sb.pairplot(data,corner=True)
# plt.show()

# mean = data["Mental-Fitness"].mean()
# mean

# Pie = px.pie(data, values="Mental-Fitness", names='Year')
# Pie.show()

# fig = px.line(data, x='Year', y = 'Mental-Fitness', color = 'Country', markers=True ,color_discrete_sequence=['red','blue'], template='plotly_dark')
# fig.show()

# data.info()

# lab = LabelEncoder()
# for i in data.columns:
#   if data[i].dtype == 'object':
#     data[i] = lab.fit_transform(data[i])

# data.shape

"""#Split Data"""

x = data.drop('Mental-Fitness',axis=1)
y = data['Mental-Fitness']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y , test_size=.20, random_state=2)

print("xtrain =", xtrain.shape)
print("xtest =", xtest.shape)
print()
print("ytrain =", ytrain.shape)
print("ytest =", ytest.shape)

"""#Model Training

## **1)Linear Regression**
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()

print(xtrain.shape)
print(ytrain.shape)
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

"""## **2)Random Forest Regression**"""

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

"""## **3)SVM Regression**"""

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

"""#Evaluation"""

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

"""#Result"""

print("------  Welcome to Mental Fitness Tracker ------")
print("------ Please Enter the Following Details ------ \n")

Country= lab.fit_transform([input('Enter Your country Name: ')])
Year= int(input("Enter the Year:"))
Schizophrenia = (float(input("Schizophrenia rate in %: ")))
Bipolar= (float(input("Bipolar disorder rate in %: ")))
Eating= (float(input("Eating disorder rate in %: ")))
Anxiety= (float(input("Anxiety rate in %: ")))
Drug_use= (float(input("Drug Usage rate in per year %: ")))
Depression= (float(input("Depression rate in %: ")))
Alcohol= (float(input("Alcohol Consuming rate per year in %: ")))

prediction= rf.predict([[Country,Year,Schizophrenia,Bipolar,Eating,Anxiety,Drug_use,Depression,Alcohol]])
print("Your Mental Fitness is {}%".format(prediction*10))
