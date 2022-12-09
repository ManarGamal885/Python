import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score,cross_val_predict
#liberaries plotting
from ipywidgets import interact, interactive, fixed, interact_manual
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv'
df =pd.read_csv(path)
df.to_csv('module_5_auto.csv')
#getting numeric data from the data
df = df._get_numeric_data()
# print(df.head())
#function for plotting
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
   def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()


#PART1 :TRAINING & TESTING
#we will separte the target var from the dataset so we can split it away from it
y_data = df['price']
x_data = df.drop('price',axis=1)
#no we will split our data using the train split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.10, random_state=3)
print("number of test samples : ", x_test.shape)
print("number of training samples : ", x_train.shape)

lre=LinearRegression()
# We fit the model using the feature "horsepower":
lre.fit(x_train[['horsepower']], y_train)
# Let's calculate the R^2 on the test data:
test = lre.score(x_test[['horsepower']], y_test)
train= lre.score(x_train[['horsepower']], y_train)
# We can see the R^2 is much smaller using the test data compared to the training data.
print(test," ", train)

#CROSS-VALIDATION SCORE
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
print(Rcross)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
# We input the object, the feature "horsepower"  , and the target data y_data. The parameter 'cv' determines the 
# number of folds. In this case, it is 4. We can produce an output:
yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
yhat[0:5]