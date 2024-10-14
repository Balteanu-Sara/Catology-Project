import pandas as pd

df = pd.read_excel('../Data/cats_data.xlsx')

print(df.info())
print(df.describe(include='all'))

missing_values = df.isnull().sum()

duplicates = df[df.duplicated()]

print("Valori lipsă:\n", missing_values)
print("Instanțe duplicate:\n", duplicates)

class_counts = df['Race'].value_counts()
print(class_counts)