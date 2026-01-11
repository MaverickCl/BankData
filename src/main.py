import pandas as pd

df = pd.read_csv("data/raw/bank.csv", sep=";")
print(df.head())
print(df.shape)
