import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
## 1) Importing data from module

path='C:\Users\msii\OneDrive\Desktop\automobileEDA.csv'
df =pd.read_csv(path)
print(df.head(5))


## 2) Analyzing Individual Feature Patterns Using Visualization


    #How to choose the right visualization method?
    #1
print(df['peak-rpm'].dtypes)
    
    #if all
print(df.corr())
    #For example, we can calculate the correlation between 4 columns using the method "corr":
print(df[['bore','stroke','compression-ratio','horsepower']].corr())
    # get correlation between element wise
print(df[['bore','stroke']].corr())
     # get correlation between element as a value
print(df['bore'].corr(df['stroke']))
    #Let's find the scatterplot of "engine-size" and "price"
    # (scatterplot:are used to plot data points on a horizontal and a vertical axis in the attempt
    # to show how much one variable is affected by anothe)
sns.regplot(x="engine-size", y="price", data=df)
    #plt.ylim(bottom,top) used to get or set the y-limits of the current axes.
    #bottom:This parameter is used to set the ylim to bottom
    #top:This parameter is used to set the ylim to top.
plt.ylim(0,)
# plt.show()
   #We can examine the correlation between 'engine-size' and 'price' and see that it's approximately 0.87.
print(df[['engine-size','price']].corr())

print(df['highway-mpg'].corr(df['price']))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)
# plt.show()

print(df['stroke'].corr(df['price']))
sns.regplot(x="stroke", y="price", data=df)
plt.ylim(0,)
# plt.show()

    #CATEGORICAL VARIABLES VISUALIZATION using boxplot
sns.boxplot(x="body-style", y="price", data=df)
plt.ylim(0,)
# plt.show()
sns.boxplot(x="engine-location", y="price", data=df)
plt.ylim(0,)
plt.show()

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.ylim(0,)
plt.show()


## 3)Descriptive Statistical Analysis


    #1-describe for continus dt
df.describe(include=['object'])
    #2-value count ->(how many units of each characteristic/variable we have)
print(df['drive-wheels'].value_counts())
    #We can convert the series to a dataframe as follows:
df['drive-wheels'].value_counts().to_frame()
    #Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" 
    # and rename the column 'drive-wheels' to 'value_counts'.
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts
print(df['engine-location'].value_counts().to_frame())


## 4)Basics of Grouping


# let's group by the variable "drive-wheels". 
# We see that there are 3 different categories of drive wheels.
# print(df['drive-wheels'].unique())
    # df_group_one = df[['drive-wheels','body-style','price']]
    # df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
    # df_group_one

df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
print(grouped_test1)

#This grouped data is much easier to visualize when it is made into a 
# pivot table. A pivot table is like an Excel spreadsheet, with one variable along the column and
# another along the row. We can convert the dataframe to a pivot table using the method 
# "pivot" to create a pivot table from the groups.

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot
print(grouped_pivot)

#using heat map to visualize data(between Body Style vs Price)
    # plt.pcolor(grouped_pivot,cmap='RdBu')
    # plt.colorbar()
    # plt.show()
#The default labels convey no useful information to us. Let's change that:
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()

## 5) Correlation and Causation

##Let's calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price
pearson_coef, p_value=stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

## 6) ANOVA
#If our price variable is strongly correlated with the variable we are 
# analyzing, we expect ANOVA to return a sizeable F-test score and a small p-value.
# ANOVA analyzes the difference between different groups of the same variable, 
#To see if different types of 'drive-wheels' impact 'price', we group the data.
grouped_test2= df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)
df_gptest

#We can use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value.

#ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'],grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])
 
print( "ANOVA results: F=", f_val, ", P =", p_val)   