import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('../Data/cats_data1.xlsx', sheet_name='Data')

df["Sexe"] = df["Sexe"].map({"NSP": 0, "F": 1, "M": 2})
df["Age"] = df["Age"].map({"Moinsde1": 0, "1a2": 1, "2a10": 2, "Plusde10": 3})
df["Race"] = df["Race"].map({"BEN": 1, "SBI": 2, "BRI": 3, "CHA": 4, "EUR": 5, "MCO": 6, "PER": 7, "RAG": 8, "SAV": 9,
                "SPH": 10, "ORI": 11, "TUV": 12, "Autre": 0, "NSP": -1, "NR": -2})
df["Nombre"] = df["Nombre"].map(lambda n: int(n) if n != "Plusde5" else 6)
df["Logement"] = df["Logement"].map({"ASB": 1, "AAB": 2, "ML": 3, "MI": 4})
df["Zone"] = df["Zone"].map({"U": 1, "PU": 2, "R": 3})
df["Abondance"] = df["Abondance"].map(lambda a: int(a) if a != "NSP" else 0)


sns.set(style="whitegrid")

# 1. Distribuția raselor de pisici (Bar Chart)
for col in df.columns:
    if not (str(col) == 'Row.names' or str(col) == 'Horodateur' or str(col) == 'Plus'):
        plt.figure(figsize=(8, 7))
        df[col].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Distribuția ' + col)
        plt.xlabel(col)
        plt.ylabel('Număr de pisici')
        plt.xticks(rotation=0)
        plt.show()