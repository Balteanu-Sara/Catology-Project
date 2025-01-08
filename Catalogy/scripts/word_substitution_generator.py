import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, sent_tokenize
import random
import langid

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


def detect_language(text):
    if text is None or len(text.strip()) == 0:
        raise ValueError("The provided text is empty or None.")

    language, confidence = langid.classify(text)
    return language


def generate_similar_text(text, replace_ratio=0.5):
    words = word_tokenize(text)
    stopwords = set(nltk.corpus.stopwords.words('english'))
    meaningful_words = []
    for w in words:
        if w.lower() not in stopwords and w.isalpha():
            meaningful_words.append(w)
    num_to_replace = max(1, int(len(meaningful_words) * replace_ratio))

    replacements = {}
    for word in random.sample(meaningful_words, num_to_replace):
        synsets = wn.synsets(word)

        if synsets:
            synonyms = set()
            hypernyms = set()
            antonyms = set()

            for synset in synsets:
                for lemma in synset.lemmas():
                    synonyms.add(lemma.name())

                    for antonym in lemma.antonyms():
                        antonyms.add(f"not {antonym.name()}")

                for hypernym in synset.hypernyms():
                    for hypernym_lemma in hypernym.lemmas():
                        hypernyms.add(hypernym_lemma.name())

            replacement = random.choice(list(synonyms | hypernyms | antonyms))
            replacements[word] = replacement

    similar_text = ' '.join([replacements.get(w, w) for w in words])
    replacements = replacements
    return similar_text, replacements


def main():
    file_path = "../Data/text"

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        language = detect_language(content)
        print(f"Detected language: {language}")

        similar_text, replacements = generate_similar_text(content)
        print("\nOriginal text:")
        print(content)
        print("\nGenerated similar text:")
        print(similar_text)
        print("\nReplacements made:")
        print(replacements)

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
