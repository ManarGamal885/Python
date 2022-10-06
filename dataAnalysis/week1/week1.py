import pandas as pd
import numpy as np

##initializing the path for the data from web or locally
path1 ="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

path ="F:\courses\dataAnalysis\imports-85.data"
##setting up the header of the data i the csv file
df =pd.read_csv(path, header=None)

print("The First 5 row of the dataframe")
print(df.head(5))
##changing the headers of the rows
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

print("headers\n", headers)

##printing the first 10 couluns
df.columns = headers
print(df.head(10))

##replacing the nan with does not exist
df1=df.replace('not exist',np.NaN)

##dropping the missing values from the price col
df=df1.dropna(subset=["price"], axis=0)
df.head(20)

##saving the database(where index = False means the row names will not be written.)
df.to_csv("automobile.csv", index=False)

#prinitng the data types of the database
print(df.dtypes)

#describing the database
print(df.describe(include="all"))

#Another method we can use to check our dataset 
print(df.info())