import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_excel('../Data/cats_data1.xlsx', sheet_name='Data')
code_df = pd.read_excel('../Data/cats_data1.xlsx', sheet_name='Code')
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

'''mapping_dict = {}
for attr in code_df['Variable'].unique():
    if attr == 'Nombre':  # Skip mapping for the 'nombre' variable
        continue

    # Create a dictionary for each attribute, splitting values and meanings
    attr_mapping = {}
    for _, row in code_df[code_df['Variable'] == attr].iterrows():
        # Check if 'Values' is a string before splitting
        if isinstance(row['Values'], str):
            values = row['Values'].split('/')  # Split using '/'
            meanings = row['Meaning']  # No change needed

            meanings_cleaned = re.sub(r'\(.*?\)', '', meanings)  # Remove parentheses content
            meanings_cleaned = meanings_cleaned.split('/')  # Split meanings

            # Clean up the meanings: Strip extra spaces
            meanings = [meaning.strip() for meaning in meanings_cleaned if meaning.strip()]

            # Create a mapping from each value to its corresponding meaning
            for value, meaning in zip(values, meanings):
                attr_mapping[value.strip()] = meaning.strip()  # Remove extra spaces
    mapping_dict[attr] = attr_mapping

for col in df.columns:
    if col not in ['Row.names', 'Horodateur', 'Plus']:
        plt.figure(figsize=(10, 8))

        # Check if the column has a mapping
        if col in mapping_dict:
            # Use the mapping for x-axis labels and replace coded values with their meanings
            # Convert values to string to handle numeric values in the DataFrame correctly
            mapped_values = df[col].astype(str).map(mapping_dict[col])
            # Count occurrences of each mapped value
            mapped_value_counts = mapped_values.value_counts()

            # Plot with mapped values as x-ticks
            mapped_value_counts.plot(kind='bar', color='skyblue')
            plt.title('Distribuția ' + col)
            plt.xlabel(col)
            plt.ylabel('Număr de pisici')
            plt.xticks(rotation=45)
            plt.show()
        else:
            # Plot categorical data without mapping
            df[col].value_counts().plot(kind='bar', color='skyblue')
            plt.title('Distribuția ' + col)
            plt.xlabel(col)
            plt.ylabel('Număr de pisici')
            plt.xticks(rotation=45)
            plt.show()'''

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
'''
# 2. Histogramă pentru distribuția comportamentelor (Timide, Calme, Effrayé)
plt.figure(figsize=(10, 6))
df[['Timide', 'Calme', 'Effrayé']].hist(bins=5, color='skyblue', alpha=0.7, figsize=(12, 8))
plt.suptitle('Distribuția atribute comportamentale')
plt.show()

# 3. Boxplot pentru atributele comportamentale (Effrayé în funcție de rasă)
plt.figure(figsize=(12, 6))
sns.boxplot(x='Race', y='Effrayé', data=df, palette='Set3')
plt.title('Distribuția valorii Effrayé în funcție de rasă')
plt.xticks(rotation=45)
plt.show()

# 4. Heatmap pentru corelația între atribute
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Corelația între atributele numerice')
plt.show()'''