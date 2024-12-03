from math import floor

import duckdb
import pandas as pd
from random import randrange, uniform

from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score


def SMOTE(sample: pd.DataFrame , N: int, k: int) -> np.array:
    T, num_attrs = sample.shape
    # If N is less than 100%, randomize the minority class samples as only a random percent of them will be SMOTEd
    if N < 100:
        T = round(N / 100 * T)
        N = 100
    # The amount of SMOTE is assumed to be in integral multiples of 100
    N = int(N / 100)
    synthetic = np.zeros([T * N, num_attrs])
    new_index = 0
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(sample.to_numpy())

    def populate(N, i, nnarray):
        nonlocal new_index
        nonlocal synthetic
        nonlocal sample
        while N != 0:
            nn = randrange(1, k + 1)
            for attr in range(num_attrs):
                dif = sample.iloc[nnarray[nn]].iloc[attr] - sample.iloc[i].iloc[attr]
                gap = uniform(0, 1)
                synthetic[new_index][attr] = sample.iloc[i].iloc[attr] + gap * dif
            new_index += 1
            N -= 1

    for i in range(T):
        nnarray = nbrs.kneighbors(sample.iloc[i].values.reshape(1, -1), return_distance=False)[0]
        populate(N, i, nnarray)

    return synthetic


df = pd.read_excel('../Data/cats_data.xlsx', sheet_name='Data')
df.drop(columns=["Row.names", "Horodateur", "Plus"],inplace=True)

df["Sexe"] = df["Sexe"].map({"NSP": 0, "F": 1, "M": 2})
df["Age"] = df["Age"].map({"Moinsde1": 0, "1a2": 1, "2a10": 2, "Plusde10": 3})
df["Race"] = df["Race"].map({"BEN": 1, "SBI": 2, "BRI": 3, "CHA": 4, "EUR": 5, "MCO": 6, "PER": 7, "RAG": 8, "SAV": 9,
                "SPH": 10, "ORI": 11, "TUV": 12, "Autre": 0, "NSP": -1, "NR": -2})
df["Nombre"] = df["Nombre"].map(lambda n: int(n) if n != "Plusde5" else 6)
df["Logement"] = df["Logement"].map({"ASB": 1, "AAB": 2, "ML": 3, "MI": 4})
df["Zone"] = df["Zone"].map({"U": 1, "PU": 2, "R": 3})
df["Abondance"] = df["Abondance"].map(lambda a: int(a) if a != "NSP" else 0)


unspecified_entries = duckdb.query("SELECT COUNT(Race) as cnt FROM df WHERE Race<0").to_df()["cnt"][0]
remaining = duckdb.query("SELECT Race,COUNT(Race) FROM df GROUP BY Race HAVING Race>=0").to_df()
total_specified = len(df) - unspecified_entries
races, weights = zip(*[(i, j / total_specified) for i, j in remaining.itertuples(index=False, name=None)])
for i in df[df["Race"] == -2].index:
    df.loc[i, "Race"] = np.random.choice(races, 1, p=weights)
for i in df[df["Race"] == -1].index:
    df.loc[i, "Race"] = np.random.choice(races, 1, p=weights)
print(duckdb.query("SELECT Race,COUNT(Race) FROM df GROUP BY Race").to_df())

X = df.drop(columns=["Race"])
y = df["Race"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(recall_score(y_test, y_pred, average="weighted"))

max_count = duckdb.query("SELECT MAX(cnt) as m FROM (SELECT COUNT(Race) as cnt FROM df GROUP BY Race)").to_df()["m"][0]
majority_entry = duckdb.query("SELECT Race,COUNT(Race) as cnt FROM df GROUP BY Race HAVING cnt=" + str(max_count)).to_df()["Race"][0]
augmented_df = df[df["Race"] == majority_entry]
for race in range(13):
    race_count = duckdb.query("SELECT Race,COUNT(Race) as cnt FROM df GROUP BY Race HAVING Race=" + str(race)).to_df()["cnt"][0]
    if race_count * 2 > max_count:
        continue
    minority = df[df["Race"] == race].drop(columns="Race")
    target_N = (floor(max_count / race_count) - 1) * 100
    synthetic = SMOTE(minority, N=target_N, k=5)
    synthetic_df = pd.DataFrame(synthetic, columns=minority.columns)
    combined_minority_df = pd.concat([minority, synthetic_df])
    combined_minority_df["Race"] = race
    augmented_df = pd.concat([combined_minority_df, augmented_df])

print(duckdb.query("SELECT Race,COUNT(Race) FROM augmented_df GROUP BY Race").to_df())
X = augmented_df.drop(columns=["Race"])
y = augmented_df["Race"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(recall_score(y_test, y_pred, average="weighted"))

#augmented_df.to_excel("cats_data_aug.xlsx")