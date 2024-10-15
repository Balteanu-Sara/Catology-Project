import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_excel('../Data/cats_data1.xlsx', sheet_name='Data')
df["Sexe"] = df["Sexe"].map({"F": 0, "M": 1})
df["Age"] = df["Age"].map({"Moinsde1": 0.5, "1a2": 1.5, "2a10": 6, "Plusde10": 12.5})
df["Race"] = df["Race"].map({"BEN": 1, "SBI": 2, "BRI": 3, "CHA": 4, "EUR": 5, "MCO": 6, "PER": 7, "RAG": 8, "SAV": 9,
                "SPH": 10, "ORI": 11, "TUV": 12, "Autre": 0, "NSP": -1, "NR": -2})
df["Nombre"] = df["Nombre"].map(lambda n: int(n) if n != "Plusde5" else 6)
df["Logement"] = df["Logement"].map({"ASB": 1, "AAB": 2, "ML": 3, "MI": 4})
df["Zone"] = df["Zone"].map({"U": 1, "PU": 2, "R": 3})
df["Ext"] = df["Ext"].map({0: 0, 1: 0.5, 2: 3, 3: 14.5, 4: 24})
df["Obs"] = df["Obs"].map({0: 0, 1: 0.5, 2: 3, 3: 6.5})
df["Abondance"] = df["Abondance"].map(lambda a: int(a) if a != "NSP" else 0)
df["PredOiseau"] = df["PredOiseau"].map({0: 0, 1: 3, 2: 7.5, 3: 24, 4: 52})
df["PredMamm"] = df["PredMamm"].map({0: 0, 1: 3, 2: 7.5, 3: 24, 4: 52})


sns.set(style="whitegrid")

# 1. Distribuția raselor de pisici (Bar Chart)
for col in df.columns:
    if not (str(col) == 'Row.names' or str(col) == 'Horodateur' or str(col) == 'Plus'):
        plt.figure(figsize=(8, 7))
        df[col].value_counts().plot(kind='bar', color='skyblue')
        plt.title('Distribuția ' + col)
        plt.xlabel(col)
        plt.ylabel('Număr de pisici')
        plt.xticks(rotation=45)
        plt.show()