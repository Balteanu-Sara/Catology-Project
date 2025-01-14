from rake_nltk import Rake
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

attribute_mapping = {
    "Sex": {0: "Unspecified", 1: "Female", 2: "Male"},
    "Age": {0: "Less than 1 year", 1.5: "1-2 years", 6: "2-10 years", 10: "More than 10 years"},
    "Number": {1: "One cat", 2: "Two cats", 3: "Three cats", 4: "Four cats", 5: "Five cats", 6: "More than five cats"},
    "Accommodation": {1: "Apartment without balcony", 2: "Apartment with balcony or terrace", 3: "House in a subdivision", 4: "Individual house"},
    "Area": {1: "Urban", 2: "Periurban", 3: "Rural"},
    "Ext": {0: "None", 1: "Limited (less than one hour)", 2: "Moderate (1 to 5 hours)", 3: "Long (more than 5 hours)", 4: "All the time (come back just to eat)"},
    "Obs": {0: "None", 1: "Limited (less than one hour)", 2: "Moderate (1 to 5 hours)", 3: "Long (more than 5 hours)"},
    "Shy": {1: "Very shy", 2: "Shy", 3: "Neutral", 4: "Not shy", 5: "Very not shy"},
    "Calm": {1: "Very calm", 2: "Calm", 3: "Neutral", 4: "Restless", 5: "Very restless"},
    "Afraid": {1: "Very afraid", 2: "Afraid", 3: "Neutral", 4: "Not afraid", 5: "Very not afraid"},
    "Clever": {1: "Not clever", 2: "Slightly clever", 3: "Clever", 4: "Very clever", 5: "Extremely clever"},
    "Vigilant": {1: "Very vigilant", 2: "Vigilant", 3: "Neutral", 4: "Not vigilant", 5: "Very not vigilant"},
    "Persevering": {1: "Not persevering", 2: "Slightly persevering", 3: "Persevering", 4: "Very persevering"},
    "Affectionate": {1: "Not affectionate", 2: "Slightly affectionate", 3: "Affectionate", 4: "Very affectionate"},
    "Friendly": {1: "Not friendly", 2: "Slightly friendly", 3: "Friendly", 4: "Very friendly"},
    "Lonely": {1: "Very lonely", 2: "Lonely", 3: "Neutral", 4: "Not lonely", 5: "Very not lonely"},
    "Brutal": {1: "Not brutal", 2: "Slightly brutal", 3: "Brutal", 4: "Very brutal"},
    "Dominant": {1: "Not dominant", 2: "Slightly dominant", 3: "Dominant", 4: "Very dominant"},
    "Aggressive": {1: "Not aggressive", 2: "Slightly aggressive", 3: "Aggressive", 4: "Very aggressive"},
    "Impulsive": {1: "Not impulsive", 2: "Slightly impulsive", 3: "Impulsive", 4: "Very impulsive"},
    "Predictable": {1: "Very unpredictable", 2: "Unpredictable", 3: "Neutral", 4: "Predictable", 5: "Very predictable"},
    "Distracted": {1: "Not distracted", 2: "Slightly distracted", 3: "Distracted", 4: "Very distracted"},
    "Abundance": {1: "Low", 2: "Moderate", 3: "High", 0: "I don't know"},
    "PredBird": {1: "Never", 2: "Rarely (1 to 5 times a year)", 3: "Sometimes (5 to 10 times a year)", 4: "Often (1 to 3 times a month)", 5: "Very often (once a week or more)"},
    "PredMamm": {1: "Never", 2: "Rarely (1 to 5 times a year)", 3: "Sometimes (5 to 10 times a year)", 4: "Often (1 to 3 times a month)", 5: "Very often (once a week or more)"},
    "Breed": {1: "Bengal", 2: "Birman", 3: "British Shorthair", 4: "Chartreux", 5: "European", 6: "Maine Coon", 7: "Persian", 8: "Ragdoll", 9: "Savannah", 10: "Sphynx", 11: "Siamese", 12: "Turkish Angora", 0: "Other", -1: "Not specified", -2: "No breed"}
}

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def extract_keywords(text):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()

def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

def encode_keywords(keywords, model):
    return model.encode(keywords, convert_to_tensor=True)

def encode_attributes(attributes, model):
    return model.encode(attributes, convert_to_tensor=True)

def calculate_similarities(keywords, keyword_embeddings, attribute_embeddings, attribute_values, attributes, model):
    results = {}
    for i, keyword in enumerate(keywords):
        similarities = cosine_similarity(
            keyword_embeddings[i].reshape(1, -1),
            attribute_embeddings
        )[0]
        best_attribute_idx = np.argmax(similarities)
        best_attribute = attributes[best_attribute_idx]
        value_embeddings = model.encode(attribute_values[best_attribute], convert_to_tensor=True)
        value_similarities = cosine_similarity(
            keyword_embeddings[i].reshape(1, -1),
            value_embeddings
        )[0]
        best_value_idx = np.argmax(value_similarities)
        best_value = attribute_values[best_attribute][best_value_idx]
        results[keyword] = (best_attribute, best_value_idx + 1, best_value)
    return results

def assign_final_values(results, attribute_mapping):
    final_values = {key: None for key in attribute_mapping.keys()}
    for keyword, (attribute, value_key, value) in results.items():
        if final_values[attribute] is None or value_key > final_values[attribute]:
            final_values[attribute] = value_key
    for attribute, default_values in attribute_mapping.items():
        if final_values[attribute] is None:
            final_values[attribute] = list(default_values.keys())[0]
    return final_values

def create_final_dict(final_values, attribute_mapping):
    return {
        attribute: {
            "value": final_values[attribute],
            "description": attribute_mapping[attribute].get(final_values[attribute], "Unknown")
        }
        for attribute in attribute_mapping.keys()
    }

def save_input_data(input_data, file_name="input_data.pkl"):
    with open(file_name, "wb") as f:
        pickle.dump(input_data, f)

def main(file_path):
    text = load_text(file_path)
    keywords = extract_keywords(text)

    model = load_model()
    attributes = list(attribute_mapping.keys())
    attribute_values = {key: list(value.values()) for key, value in attribute_mapping.items()}
    keyword_embeddings = encode_keywords(keywords, model)
    attribute_embeddings = encode_attributes(attributes, model)

    results = calculate_similarities(keywords, keyword_embeddings, attribute_embeddings, attribute_values, attributes,
                                     model)

    final_values = assign_final_values(results, attribute_mapping)

    final_dict = create_final_dict(final_values, attribute_mapping)

    print(final_dict)

    if 'Breed' in final_dict:
        del final_dict['Breed']

    input_data = [details['value'] for attribute, details in final_dict.items()]

    save_input_data(input_data)
