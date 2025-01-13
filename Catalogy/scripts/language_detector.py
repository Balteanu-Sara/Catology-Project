import langid
import re
from collections import Counter

file_path = "../Data/text"

try:
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    num_chars = len(content)
    text = content.lower()
    words = re.findall(r'\b\w+\b', text)
    word_count = Counter(words)
    num_words = len(words)

    print(f"Number of characters: {num_chars}")
    print(f"Number of words: {num_words}")
    print("Word frequencies:")
    for word, count in word_count.most_common():
        print(f"{word}: {count}")

    language, _= langid.classify(content)
    print(f"Detected language: {language}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")


