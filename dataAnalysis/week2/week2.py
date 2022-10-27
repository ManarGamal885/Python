import pandas as pd
import numpy as np 
filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df = pd.read_csv(filename, names=headers)

#DEALING WITH DATA

    #Identify Missing Data
    #convert ? to NaN
df.replace("?", np.nan, inplace=True)
'''print(df.head(5))'''
missing_data = df.isnull()
'''print(missing_data.head(5))'''
    #Count Missing Values
for col in missing_data.columns.values.tolist():
    print(col)
    print(missing_data[col].value_counts())
    print("")

    #Calculate the mean value for the "normalized-losses" column
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)

    #Replacing the nan with mean 
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)
df["bore"].replace(np.nan, avg_bore, inplace=True)

    #to see which values are present in a particular column, we can use the ".value_counts()" method:
df['num-of-doors'].value_counts()
    #We can also usse the ".idxmax()" method to calculate the most common type automatically:
df['num-of-doors'].value_counts().idxmax()

    #replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)

    # simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

    # reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)
df.head()