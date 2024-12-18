import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('../Data/cats_data1.xlsx')

df["Sexe"] = df["Sexe"].map({"NSP": 0, "F": 1, "M": 2})
df["Age"] = df["Age"].map({"Moinsde1": 0, "1a2": 1, "2a10": 2, "Plusde10": 3})
df["Race"] = df["Race"].map({"BEN": 1, "SBI": 2, "BRI": 3, "CHA": 4, "EUR": 5, "MCO": 6, "PER": 7, "RAG": 8, "SAV": 9,
                "SPH": 10, "ORI": 11, "TUV": 12, "Autre": 0, "NSP": -1, "NR": -2})
df["Nombre"] = df["Nombre"].map(lambda n: int(n) if n != "Plusde5" else 6)
df["Logement"] = df["Logement"].map({"ASB": 1, "AAB": 2, "ML": 3, "MI": 4})
df["Zone"] = df["Zone"].map({"U": 1, "PU": 2, "R": 3})
df["Abondance"] = df["Abondance"].map(lambda a: int(a) if a != "NSP" else 0)

print(df.info())

missing_values = df.isnull().sum()

duplicates = df[df.duplicated()]

print("Valori lipsă:\n", missing_values)
print("Instanțe duplicate:\n", duplicates)

class_counts = df['Race'].value_counts()
print(class_counts)

for col in df.columns:
    if col not in ['Row.names', 'Horodateur', 'Plus']:
        print(f"\nAttribute: {col}")
        print(df[col].value_counts())


def distinct_values_per_class(df, class_column):
    attributes = df.columns.difference([class_column, 'Row.names', 'Horodateur', 'Plus'])

    for class_value in df[class_column].unique():
        if class_value not in ['Row.names', 'Horodateur', 'Plus']:

            print(f"\n----------{class_column}: {class_value} \n")
            class_group = df[df[class_column] == class_value]

            for attr in attributes:
                print(f"\nAtribut: {attr}")

                distinct_values = class_group[attr].value_counts()
                for value, count in distinct_values.items():
                    print(f"  Valoare: {value}, Frecvență: {count}")


distinct_values_per_class(df, 'Race')

sns.set(style="whitegrid")
plt.figure(figsize=(8, 7))
correlation_matrix = df.drop(['Plus', 'Row.names', 'Horodateur'],axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 5})
plt.title('Corelația între atributele numerice')
plt.show()

