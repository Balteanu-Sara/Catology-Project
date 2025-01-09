from rake_nltk import Rake
from keytotext import pipeline

file_path = "../Data/text"
rake_nltk_var = Rake()
rake_nltk_var.extract_keywords_from_text(open(file_path, "r").read())
keyword_extracted = rake_nltk_var.get_ranked_phrases()

nlp = pipeline("mrm8488/t5-base-finetuned-common_gen")
for i in keyword_extracted:
    print(f"keyword: {i}")
    print(nlp([i]))