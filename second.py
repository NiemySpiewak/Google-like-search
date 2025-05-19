import os
import json
def vocabulary(path):
    big_word_set = set()
    counter = 0

    print(f"Rozpoczynam wczytywanie dokumentów z katalogu: `{path}`")

    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if counter % 100 ==0:
            print(f"Przetwarzanie pliku: `{filename}`")

        with open(filepath, "r", encoding="utf-8") as f:
            doc_lines = f.readlines()

        text_start = False
        doc_text = ""

        for line in doc_lines:
            if not text_start and line.strip() == "":
                text_start = True
                continue
            if text_start:
                doc_text += line.strip() + " "

        words = doc_text.strip().split()
        big_word_set.update(words)
        counter += 1

    print(f"\n Przetworzono {counter} dokumentów.")
    print(f"Całkowita liczba unikalnych słów w korpusie: {len(big_word_set)}")

    return big_word_set


def save_vocab(vocabulary_set,output_filename):
    sorted_vocabulary = sorted(list(vocabulary_set))
    with open(output_filename, "w", encoding="utf-8") as f:
        for word in sorted_vocabulary:
            f.write(word + "\n")

def big_bag_of_words(voc_set, out):
    sorted_voc = sorted(list(voc_set))
    big_bow = {word: i for i, word in enumerate(sorted_voc)}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(big_bow, f, indent=2)
    return big_bow

def main():

    voc = vocabulary("docs")

    save_vocab(voc,"vocabulary.txt")
    print("Zapisano słownik!")
    big_bag_of_words(voc, "big_bow.json")
    print("Zapisano BagOfWords duże!")

main()