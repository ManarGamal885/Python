import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(path)

### 1)Linear Regression and Multiple Linear Regression


# 1-LINEAR REGRESSION

# " Simple Linear Regression

# we want to predict y price such that highway-mpg is predictor and price is response var
lm =LinearRegression()
lm
# X response  a intercept
# Y predictor b slope
X = df[['highway-mpg']]
Y = df[['price']]
# to get a and b values
lm.fit(X,Y)
# note that the predict function will return a list of all predicted values of 
# eq Price= 38423.31 - 821.73 x highway-mpg 
Yhat = lm.predict(X)
# ?
Yhat[0:5]   
# print(lm.intercept_)
# print(lm.coef_)
# print(Yhat)
# note that the yhat will give us a list of numbers because the highway-mpg will vary
# since Yhat = a+bx
#  Price= 38423.31 - 821.73 x highway-mpg

# 2-MULTIPLE LINEAR REGRESSION
# it contains one continus response and two or more predictor
''' Y: Response  Variable
X1 :Predictor Variable 1
X2: Predictor Variable 2
X3: Predictor Variable 3
X4: Predictor Variable 4

a: intercept
b1 :coefficients  of Variable 1
b2: coefficients  of Variable 2
b3: coefficients  of Variable 3
b4: coefficients  of Variable 4

Yhat = a + b1 X1 + b2 X2 + b3 X3 + b4 X4

'''
#predicted vars
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
L = df[['price']]
lm.fit(Z,L)
Yhat1= lm.predict(Z)
# print(Yhat1)
# print(lm.coef_)
# print(lm.intercept_)




### 2) Model Evaluation Using Visualization


'''Now that we've developed some models, how do we evaluate our models and choose the best one? 
One way to do this is by using a visualization.'''

# Regression Plot
width = 6
height = 5
plt.figure(figsize=(width,height))
sns.regplot(x="highway-mpg", y = "price", data = df)
plt.ylim(0,)
# plt.show()

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
# plt.show()

# VISUALIZATIO to take dicision

# 1- Linear regression

# Residual Plot
# A good way to visualize the variance of the data is to use a residual plot
plt.figure(figsize=(width,height))
sns.residplot(x=df['highway-mpg'],y=df['price'])
# plt.show()
# nonlinear is more approperate

# 2-Multiple Linear Regression
#  distribution plot
plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat1, hist=False, color="b", label="Fitted Values" , ax=ax1)

# red actual
# blue prediction
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

# plt.show()
# plt.close()


### 3) Polynomial regression
# is a particular case of the general linear 
# regression model or multiple linear regression models.
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']
# Let's fit the polynomial using the function polyfit, then use the function poly1d to display the polynomial function.
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)
# PlotPolly(p, x, y, 'highway-mpg')
# Here we use a polynomial of the 3rd order (cubic)
np.polyfit(x, y, 3)

# We can perform a polynomial transform on multiple features
# We create a PolynomialFeatures object of degree 2:
# we use this wen we want to make more that two vars in multiple degress
pr=PolynomialFeatures(degree=2)
pr
Z_pr=pr.fit_transform(Z)
Z.shape
Z_pr.shape

# Pipeline

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
pipe
Z = Z.astype(float)
pipe.fit(Z,y)
# Similarly,  we can normalize the data, perform a 
# transform and produce a prediction  simultaneously
ypipe=pipe.predict(Z)
ypipe[0:4]




### 4) Measures for In-Sample Evaluation


# determine how accurate the model is

# Model 1: Simple Linear Regression

#highway_mpg_fit
lm.fit(X, Y)
# Find the R^2
print('The R-square is: ', lm.score(X, Y))
# We can say that \~49.659% of the variation of the price is explained by 
# this simple linear model "horsepower_fit
# We can predict the output i.e., "yhat" using the predict 
# method, where X is the input variable:
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])
# We can compare the predicted results with the actual results:
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)



# Model 2: Multiple Linear Regression

# Let's calculate the R^2:
# fit the model 
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))
# We produce a prediction:
Y_predict_multifit = lm.predict(Z)
# We compare the predicted results with the actual results:
print('The mean square error of price and predicted value using multifit is: ', \
mean_squared_error(df['price'], Y_predict_multifit))



# Model 3: Polynomial Fit

# Let's calculate the R^2.
# We apply the function to get the value of R^2:
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
# We can also calculate the MSE:
mean_squared_error(df['price'], p(x))



### 5)Prediction and Decision Making

new_input=np.arange(1, 100, 1).reshape(-1, 1)
lm.fit(X, Y)
lm
# Produce a prediction:
yhat=lm.predict(new_input)
yhat[0:5]
# We can plot the data:
plt.plot(new_input, yhat)
plt.show()