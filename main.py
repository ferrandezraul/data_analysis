import pandas as pd

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

