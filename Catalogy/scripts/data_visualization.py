import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Încarcă setul de date
df = pd.read_excel('../Data/cats_data.xlsx')

sns.set(style="whitegrid")

# 1. Distribuția raselor de pisici (Bar Chart)
plt.figure(figsize=(10, 6))
df['Race'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribuția raselor de pisici')
plt.xlabel('Rasă')
plt.ylabel('Număr de pisici')
plt.xticks(rotation=45)
plt.show()

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
plt.show()