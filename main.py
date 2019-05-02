import pandas as pd
import numpy as np

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

# Continue lab https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/DA0101EN/edx/DA0101EN-Review-Data_Wrangling-py-v4pp.ipynb
# with data Standarization chapter



