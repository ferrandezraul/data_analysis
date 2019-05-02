import pandas as pd

df = pd.read_csv("data_sets/wine_quality/winequality-white.csv", sep=';')

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
print("Printing column 'citric acid' of the data frame")
print(df["citric acid"])

