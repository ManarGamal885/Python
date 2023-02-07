import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.utils.validation import check_is_fitted

##Prepare Data

#Task 2.1.1: Write a function named wrangle that takes a file path as an argument and returns a DataFrame.
def wrangle(filePath):
    df = pd.read_csv(filePath)
    return df
#Task 2.1.1: Use your wrangle function to create a DataFrame df from the CSV file data/buenos-aires-real-estate-1.csv.
df = wrangle("data/buenos-aires-real-estate-1.csv")
print("df shape:", df.shape)
df.head()

#Task 2.1.3: Add to your wrangle function so that the DataFrame it returns only includes apartments in Buenos Aires ("Capital Federal") that cost less than $400,000 USD. Then recreate df from data/buenos-aires-real-estate-1.csv by re-running the cells above
def wrangle(filePath):
    df = pd.read_csv(filePath)
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"]<400_000
    df = df[mask_ba & mask_apt & mask_price]
    return df
#Task 2.1.4: Create a histogram of "surface_covered_in_m2". Make sure that the x-axis has the label "Area [sq meters]" and the plot has the title "Distribution of Apartment Sizes".
plt.hist(df["surface_covered_in_m2"])
plt.xlabel("Area [sq meters]")
plt.title("Distribution of Apartment Sizes");
#Task 2.1.5: Calculate the summary statistics for df using the describe method.
df.describe()["surface_covered_in_m2"]

#remove outlayers "surface_covered_in_m2"
#Task 2.1.6: Add to your wrangle function so that it removes observations that are outliers in the "surface_covered_in_m2" column. Specifically, all observations should fall between the 0.1 and 0.9 quantiles for "surface_covered_in_m2".
def wrangle(filePath):
    df = pd.read_csv(filePath)
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"]<400_000
    df = df[mask_ba & mask_apt & mask_price]
    low, heigh = df["surface_covered_in_m2"].quantile([0.1,0.9])
    mask_area = df["surface_covered_in_m2"].between(low,heigh)
    df = df[mask_area]
    return df
#Task 2.1.7: Create a scatter plot that shows price ("price_aprox_usd") vs area ("surface_covered_in_m2") in our dataset. Make sure to label your x-axis "Area [sq meters]" and your y-axis "Price [USD]"
plt.scatter(x = df["surface_covered_in_m2"], y = df["price_aprox_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Buenos Aires: Price vs. Area")
#Task 2.1.8: Create the feature matrix named X_train, which you'll use to train your model. It should contain one feature only: ["surface_covered_in_m2"]. Remember that your feature matrix should always be two-dimensional.
features = ["surface_covered_in_m2"]
X_train = df[features]
X_train.shape
#Task 2.1.9: Create the target vector named y_train, which you'll use to train your model. Your target should be "price_aprox_usd". Remember that, in most cases, your target vector should be one-dimensional.
target = "price_aprox_usd"
y_train = df[target]
y_train.head()
y_train.shape

##Build Model
#Baseline
#Task 2.1.10: Calculate the mean of your target vector y_train and assign it to the variable y_mean.
y_mean = y_train.mean()
y_mean
#Task 2.1.11:** Create a list named `y_pred_baseline` that contains the value of `y_mean` repeated so that it's the same length at `y`.
y_pred_baseline = [y_mean]*len(y_train)
len(y_pred_baseline) == len(y_train)
#Task 2.1.12: Add a line to the plot below that shows the relationship between the observations X_train and our dumb model's predictions y_pred_baseline. Be sure that the line color is orange, and that it has the label "Baseline Model"
plt.plot(X_train["surface_covered_in_m2"],y_pred_baseline, color = "yellow", label ="Base Line Model")
plt.scatter(X_train, y_train)
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Buenos Aires: Price vs. Area")
plt.legend();
#Task 2.1.13: Calculate the baseline mean absolute error for your predictions in y_pred_baseline as compared to the true targets in y.
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean apt price", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))
#Task 2.1.14: Instantiate a LinearRegression model named model.
model = LinearRegression()
#Task 2.1.15: Fit your model to the data, X_train and y_train.
model.fit(X_train,y_train)
#Task 2.1.16: Using your model's predict method, create a list of predictions for the observations in your feature matrix X_train. Name this array y_pred_training.
y_pred_training = model.predict(X_train)
y_pred_training[:5]
#Task 2.1.17: Calculate your training mean absolute error for your predictions in y_pred_training as compared to the true targets in y_train.
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))
#Task 2.1.18: Run the code below to import your test data buenos-aires-test-features.csv into a DataFrame and generate a Series of predictions using your model. Then run the following cell to submit your predictions to the grader.
X_test = pd.read_csv("data/buenos-aires-test-features.csv")[features]
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()
#Task 2.1.19:** Extract the intercept from your model, and assign it to the variable `intercept`. 
intercept = model.intercept_
print("Model Intercept:", intercept)
assert any([isinstance(intercept, int), isinstance(intercept, float)])
#Task 2.1.20: Extract the coefficient associated "surface_covered_in_m2" in your model, and assign it to the variable coefficient.
coefficient = round(model.coef_[0], 2)
print('Model coefficient for "surface_covered_in_m2":', coefficient)
assert any([isinstance(coefficient, int), isinstance(coefficient, float)])
#Task 2.1.21: Complete the code below and run the cell to print the equation that your model has determined for predicting apartment price based on size.
print(f"apt_price ={intercept} + {coefficient} * surface_covered")
#Task 2.1.22: Add a line to the plot below that shows the relationship between the observations in X_train and your model's predictions y_pred_training. Be sure that the line color is red, and that it has the label "Linear Model".
plt.plot(X_train["surface_covered_in_m2"],model.predict(X_train),color="r",label="Linear Model")
plt.scatter(X_train, y_train)
plt.xlabel("surface covered [sq meters]")
plt.ylabel("price [usd]")
plt.legend();