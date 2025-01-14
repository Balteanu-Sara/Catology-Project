import os
import key2text_make_sentences
import neuralNet
from Catalogy.scripts import gan_generate


def ensure_backslash(path):
    normalized_path = path.replace("\\", "\\\\")
    path = normalized_path.replace("/", "\\\\")
    if path.startswith(("'", '"')) and path.endswith(("'", '"')):
        path = path[1:-1]
    return path

bool = True
commands = ["identify_breed", "describe_breed", "exit"]

while bool:
    command = input("Type a command (identify_breed | describe_breed | exit) : ").strip().lower()

    if command not in commands:
        print("The command is not valid.")
        continue

    if command == "identify_breed":
        bool1 = True
        print("Required arguments : {file_path}")

        while bool1:
            bool2 = True
            while bool2:
                file_path = input("Type file path : ").strip().lower()
                file_path = ensure_backslash(file_path)

                if not os.path.isfile(file_path):
                    print("The file given as input is not valid.")
                    continue

                bool2 = False
            key2text_make_sentences.main(file_path)
            neuralNet.main()

            bool1 = False


    if command == "describe_breed":
        gan_generate.main()

    if command == "exit":
        bool = False