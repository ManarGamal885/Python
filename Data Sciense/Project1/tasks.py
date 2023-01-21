from IPython.display import VimeoVideo
import pandas as pd
#1.1. Organizing Tabular Data in Python

#TABULAR
#adding a single observation list
house_0_list = [115910.26,128,4]
#Task1.1.1
house_0_price_m2 = house_0_list[0]/house_0_list[1]
print(house_0_price_m2)

#Task1.1.2
house_0_list.append(house_0_price_m2)
print(house_0_list)

#Task1.1.3
houses_nested_list = [
    [115910.26, 128.0, 4.0],
    [48718.17, 210.0, 3.0],
    [28977.56, 58.0, 2.0],
    [36932.27, 79.0, 3.0],
    [83903.51, 111.0, 3.0],
]
for x in houses_nested_list:
   price_m2 = x[0]/x[1]
   x.append(price_m2)
for x in houses_nested_list:
    print(x)


#DICTIONARY
house_0_dict = {
    "price_aprox_usd": 115910.26,
    "surface_covered_in_m2": 128,
    "rooms": 4,
}
print(house_0_dict)
#Task1.1.4
house_0_dict["price_per_m2"] = house_0_dict["price_aprox_usd"]/house_0_dict["surface_covered_in_m2"]
#Task1.1.5
houses_rowwise = [
    {
        "price_aprox_usd": 115910.26,
        "surface_covered_in_m2": 128,
        "rooms": 4,
    },
    {
        "price_aprox_usd": 48718.17,
        "surface_covered_in_m2": 210,
        "rooms": 3,
    },
    {
        "price_aprox_usd": 28977.56,
        "surface_covered_in_m2": 58,
        "rooms": 2,
    },
    {
        "price_aprox_usd": 36932.27,
        "surface_covered_in_m2": 79,
        "rooms": 3,
    },
    {
        "price_aprox_usd": 83903.51,
        "surface_covered_in_m2": 111,
        "rooms": 3,
    },
]
for house in houses_rowwise:
   house["price_per_m2"] = house_0_dict["price_aprox_usd"]/house_0_dict["surface_covered_in_m2"]

for house in houses_rowwise:
    print(house)

#Task1.1.6
house_prices = []
for house in houses_rowwise:
    house_prices.append(house["price_aprox_usd"])
mean_house_price = sum(house_prices) / len(house_prices)
print(mean_house_price)

#Task1.1.7
houses_columnwise = {
    "price_aprox_usd": [115910.26, 48718.17, 28977.56, 36932.27, 83903.51],
    "surface_covered_in_m2": [128.0, 210.0, 58.0, 79.0, 111.0],
    "rooms": [4.0, 3.0, 2.0, 3.0, 3.0],
}
mean_house_price = sum(houses_columnwise["price_aprox_usd"])/len(houses_columnwise["price_aprox_usd"])
print(mean_house_price)

#Task1.1.8
price = houses_columnwise["price_aprox_usd"]
area =  houses_columnwise["surface_covered_in_m2"]
price_per_m2 = []
for p, a in zip(price,area):
    price_m2 = p/a
    price_per_m2.append(price_m2)
houses_columnwise ["price_per_m2"] = price_per_m2
print(houses_columnwise)

#OVERVIEW
data = {
    "price_aprox_usd": [115910.26, 48718.17, 28977.56, 36932.27, 83903.51],
    "surface_covered_in_m2": [128.0, 210.0, 58.0, 79.0, 111.0],
    "rooms": [4.0, 3.0, 2.0, 3.0, 3.0],
}
df_houses = pd.DataFrame(data)
print(df_houses)

#1.2. Preparing Mexico Data
#Task1.2.1
df1 = pd.read_csv("data/mexico-real-estate-1.csv")
df2 = pd.read_csv("data/mexico-real-estate-2.csv")
df3 = pd.read_csv("data/mexico-real-estate-3.csv")

#Task1.2.2
#clean df1
df1.shape
#return no. of rows and no. of cols
df1.info()

#Task1.2.3
#Remove NaN values
df1.dropna(inplace = True)
df1["price_usd"] = (
    df1["price_usd"]
    .str.replace("$", "", regex = False)
    .str.replace(",", "")
    .
    astype(float)
)

#Task1.2.4
#clean df2
#Remove 'Nan'
df2.dropna(inplace = True)
#Create "Price_usd" col
df2["price_usd"] = (df2["price_mxn"] / 19).round(2)
#Drop a col
df2.drop(columns=["price_mxn"], inplace = True)
df2.head()

#Task1.2.5
#clean df3
#expand to make it in columns instade of a list
df3[["lat", "lon"]] = df3["lat-lon"].str.split(",", expand = True)
df3.head()

#Task1.2.6
df3["state"] = df3["place_with_parent_names"].str.split("|", expand = True)[2]
df3.drop(columns=["place_with_parent_names","lat-lon"], inplace = True )
df3.head()

#Task1.2.7
#axis = 0 vertically =1 horizontally
df = pd.concat([df1, df2, df3])
print(df.shape)
df.head()

#Task1.2.8
#save DF
df.to_csv("data/mexico-real-estate-clean.csv", index = False)