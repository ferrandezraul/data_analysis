import pandas as pd
import numpy as np
import matplotlib as plt
from matplotlib import pyplot

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
           "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
           "peak-rpm", "city-mpg", "highway-mpg", "price"]

df = pd.read_csv("data_sets/cars/auto.csv", sep=',', names=headers)

print("Printing data head")
print(df.head(5))

print("\n")
print("Printing data tail")
print(df.tail(5))

print("\n")
print("Printing data describe")
print(df.describe(include='all'))

print("\n")
print("Printing data types")
print(df.dtypes)

print("\n")
print("Printing data info")
print(df.info())

print("\n")
print("Printing column 'make' of the data frame")
print(df["make"])
print("\n")

print("Is ? inside the dataframe?")
print(df.isin(["?"]))
print("\n")

# Identify missing values and convert '?' to NaN
df.replace("?", np.nan, inplace=True)

print("Is now ? inside the dataframe?")
print(df.isin(["?"]))
print("\n")

# Check for missing values
missing_data = df.isnull()
print("Printing missing values")
print(missing_data.head(20))
print("\n")

# value_counts() counts the number of "True" values.
print("Printing number of missing values in each column")
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

print("\n")

# Column "normalized-losses": 41 missing data, replace them with mean
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)
print("\n")

# Column "bore": 4 missing data, replace them with mean
avg_bore = df["bore"].astype("float").mean(axis=0)
print("Average of bore:", avg_bore)
df["bore"].replace(np.nan, avg_bore, inplace=True)
print("\n")

# Column "stroke": 4 missing data, replace them with mean
avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke:", avg_stroke)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)
print("\n")

# Column "horsepower": 2 missing data, replace them with mean
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
print("\n")

# Column "peak-rpm": 2 missing data, replace them with mean
avg_peak_rpm = df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak-rpm:", avg_peak_rpm)
df['peak-rpm'].replace(np.nan, avg_peak_rpm, inplace=True)
print("\n")

# Column "price": 4 missing data, replace them with mean
avg_price = df["price"].astype("float").mean(axis=0)
print("Average price:", avg_price)
df["price"].replace(np.nan, avg_price, inplace=True)
print("\n")

# Count the number of values
print("Number of values in column 'num-of-doors' \n", df['num-of-doors'].value_counts())
print("\n")
print("Number of values in column 'make' \n", df['make'].value_counts())
print("\n")
print("Number of values in column 'engine-location' \n", df['engine-location'].value_counts())

print("\n")
print("Printing data types")
print(df.dtypes)
print("\n")

print("Convert data types to proper format")
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

print("\n")
print("Printing data types after converting them to the right format")
print(df.dtypes)
print("\n")

# Data Standarization
# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]
df['highway-L/100km'] = 235/df["highway-mpg"]
print("Printing columns city-L/100km and highway-L/100km: \n", df[["city-L/100km", "highway-L/100km"]].head())
print("\n")

print("Printing values for length, width and height before normalization: \n", df[["length", "width", "height"]].head())
print("\n")

# Data Normalization
# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()

print("Printing Normalizaed values for length, width and height: \n", df[["length", "width", "height"]].head())
print("\n")

# Binning
# In our dataset, "horsepower" is a real valued variable ranging from 48 to 288, it has 57 unique values.
# What if we only care about the price difference between cars with high horsepower,
# medium horsepower, and little horsepower (3 types)?
# Can we rearrange them into three ‘bins' to simplify analysis?
# We will use the Pandas method 'cut' to segment the 'horsepower' column into 3 bins
df["horsepower"] = df["horsepower"].astype(int, copy=True)

print("Printing df['horsepower']", df["horsepower"])

plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
plt.pyplot.show()

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower', 'horsepower-binned']].head(20)

print("Number of vehicles in each bin")
print(df["horsepower-binned"].value_counts())
print("\n")

pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
plt.pyplot.show()

# indicator variable
# An indicator variable (or dummy variable) is a numerical variable used to label categories.
# They are called 'dummies' because the numbers themselves don't have inherent meaning.
# We will use the panda's method 'get_dummies' to assign numerical values to different categories of fuel type.
# get indicator variables and assign it to data frame "dummy_variable_1"
print("Printing df['fuel-type']")
print(df["fuel-type"])
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
print("Printing dummy_variable_1 head")
print(dummy_variable_1.head())

# change column names for clarity
dummy_variable_1.rename(columns={'fuel-type-gas':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
print("Printing dummy_variable_1 head")
print(dummy_variable_1.head())

# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis=1, inplace=True)
print(df.head())

print("Printing aspiration column")
print(df["aspiration"])
dummy_variable_2 = pd.get_dummies(df["aspiration"])
print("Printing dummy_variable_2 head")
print(dummy_variable_2.head())

# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo':'aspiration-turbo'}, inplace=True)
print("Printing dummy_variable_2 head")
print(dummy_variable_2.head())

# merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "fuel-type" from "df"
df.drop("aspiration", axis=1, inplace=True)
print(df.head())

df.to_csv('clean_df.csv')





