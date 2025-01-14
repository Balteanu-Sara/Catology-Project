from sdgx.data_connectors.dataframe_connector import DataFrameConnector
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
import pandas as pd

attribute_mapping = {
    "Sex": {0: "Unspecified", 1: "Female", 2: "Male"},
    "Age": {0: "Less than 1 year", 1: "1-2 years", 2: "2-10 years", 3: "More than 10 years"},
    "Number": {1: "One cat", 2: "Two cats", 3: "Three cats", 4: "Four cats", 5: "Five cats", 6: "More than five cats"},
    "Accommodation": {1: "Apartment without balcony", 2: "Apartment with balcony or terrace", 3: "House in a subdivision", 4: "Individual house"},
    "Area": {1: "Urban", 2: "Periurban", 3: "Rural"},
    "Time spent outdoors": {0: "None", 1: "Limited (less than one hour)", 2: "Moderate (1 to 5 hours)", 3: "Long (more than 5 hours)", 4: "All the time (come back just to eat)"},
    "Time spent with owner": {0: "None", 1: "Limited (less than one hour)", 2: "Moderate (1 to 5 hours)", 3: "Long (more than 5 hours)"},
    "Shy": {1: "Very shy", 2: "Shy", 3: "Neither shy nor confident", 4: "Confident", 5: "Very confident"},
    "Calm": {1: "Very calm", 2: "Calm", 3: "Neither calm nor restless", 4: "Restless", 5: "Very restless"},
    "Afraid": {1: "Very afraid", 2: "Afraid", 3: "Neither afraid nor unafraid", 4: "Not unafraid", 5: "Very unafraid"},
    "Clever": {1: "Not clever", 2: "Slightly clever", 3: "Clever", 4: "Very clever", 5: "Extremely clever"},
    "Vigilant": {1: "Very vigilant", 2: "Vigilant", 3: "Not particularly vigilant", 4: "Oblivious", 5: "Very oblivious"},
    "Persevering": {1: "Not persevering", 2: "Slightly persevering", 3: "Persevering", 4: "Very persevering", 5: "Extremely persevering"},
    "Affectionate": {1: "Not affectionate", 2: "Slightly affectionate", 3: "Affectionate", 4: "Very affectionate", 5: "Extremely affectionate"},
    "Friendly": {1: "Not friendly", 2: "Slightly friendly", 3: "Friendly", 4: "Very friendly", 5: "Extremely friendly"},
    "Lonely": {1: "Very lonely", 2: "Lonely", 3: "Not too sociable, not too unsociable", 4: "Sociable", 5: "Very sociable"},
    "Brutal": {1: "Not brutal", 2: "Slightly brutal", 3: "Brutal", 4: "Very brutal", 5: "Extremely brutal"},
    "Dominant": {1: "Not dominant", 2: "Slightly dominant", 3: "Dominant", 4: "Very dominant", 5: "Extremely dominant"},
    "Aggressive": {1: "Not aggressive", 2: "Slightly aggressive", 3: "Aggressive", 4: "Very aggressive", 5: "Extremely aggressive"},
    "Impulsive": {1: "Not impulsive", 2: "Slightly impulsive", 3: "Impulsive", 4: "Very impulsive", 5: "Extremely impulsive"},
    "Predictable": {1: "Very unpredictable", 2: "Unpredictable", 3: "Neutral", 4: "Predictable", 5: "Very predictable"},
    "Distracted": {1: "Not distracted", 2: "Slightly distracted", 3: "Distracted", 4: "Very distracted", 5: "Extremely distracted"},
    "Abundance": {1: "Low", 2: "Moderate", 3: "High", 0: "I don't know"},
    "Catches birds": {0: "Never", 1: "Rarely (1 to 5 times a year)", 2: "Sometimes (5 to 10 times a year)", 3: "Often (1 to 3 times a month)", 4: "Very often (once a week or more)"},
    "Catches mice": {0: "Never", 1: "Rarely (1 to 5 times a year)", 2: "Sometimes (5 to 10 times a year)", 3: "Often (1 to 3 times a month)", 4: "Very often (once a week or more)"},
    "Breed": {1: "Bengal", 2: "Birman", 3: "British Shorthair", 4: "Chartreux", 5: "European", 6: "Maine Coon", 7: "Persian", 8: "Ragdoll", 9: "Savannah", 10: "Sphynx", 11: "Siamese", 12: "Turkish Angora", 0: "Other", -1: "Not specified", -2: "No breed"}
}

data = pd.read_excel("../Data/cats_data_en.xlsx", sheet_name="Data")

# Create data connector for csv file
data_connector = DataFrameConnector(df=data)

# Initialize synthesizer, use CTGAN model
synthesizer = Synthesizer(
    model=CTGANSynthesizerModel(epochs=1),  # For quick demo
    data_connector=data_connector,
)

# Fit the model
synthesizer.fit()

# Sample
sampled_data = synthesizer.sample(1)
for name, values in sampled_data.items():
    print(f"{name}: {attribute_mapping[name][values[0]]}")