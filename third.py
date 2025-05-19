def BagOfWords(text):
    list_of_words = text.split()
    bag = {}
    for word in list_of_words:
        if word in bag:
            bag[word] +=1
        else:
            bag[word] = 1
    return bag


import os
import json
import numpy as np
from scipy.sparse import lil_matrix, diags, save_npz

# --- Funkcje pomocnicze ---
def load_processed_documents_from_files(dir):
    counter = 0
    docs_data = [] # Lista krotek (filename, content)
    for filename in sorted(os.listdir(dir)):
        counter +=1
        if counter % 100 == 0:
            print(f"counter:{counter}")
        filepath = os.path.join(dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            content_started = False
            doc_text = ""
            if len(lines) > 2 and lines[0].lower().startswith("url:") and lines[1].strip() == "":
                doc_text = "".join(lines[2:]).strip()
            elif any(line.strip() == "" for line in lines):
                for line in lines:
                    if not content_started and line.strip() == "":
                        content_started = True; continue
                    if content_started: doc_text += line.strip() + " "
            else: doc_text = "".join(lines).strip()
            
            if doc_text.strip():
                docs_data.append((filename, doc_text.strip()))
    return docs_data

def load_big_bow(path):
    vocab_map = {}

    with open(path, "r", encoding="utf-8") as f:
        vocab_map = json.load(f)

    return vocab_map



def create_matrix(docs_data, big_bag):
    print(f"Rozpoczynam budowę macierzy z {len(docs_data)} dokumentów...")
    print(f"Liczba unikalnych słów (wielki worek): {len(big_bag)}")

    # lil_matrix dobra do budowy, csr_matrix do obliczeń
    matrix = lil_matrix((len(big_bag), len(docs_data)))

    document_order = []

    for doc_idx, (filename, doc_text) in enumerate(docs_data):
        print(f"\nDokument #{doc_idx+1}: {filename}")
        document_order.append(filename)

        small_bow = BagOfWords(doc_text)
        print(f"Liczba słów w BOW: {len(small_bow)}")

        filled = 0
        for word, count in small_bow.items():
            if word in big_bag:
                idx = big_bag[word]
                matrix[idx, doc_idx] = count
                filled += 1
        print(f"Wypełnione {filled} pozycji w macierzy dla tego dokumentu.")

    matrix = matrix.tocsr()
    print(f"\nKonwersja do formatu CSR zakończona.")

    with open("document_order.json", "w", encoding="utf-8") as f:
        json.dump(document_order, f, indent=2)
        print("Zapisano `document_order.json`!")

    print(f"\nGotowa macierz o wymiarach: {matrix.shape}")
    return matrix, document_order

# liczymy IDF
def IDF_calc(matrix):

    _, N = matrix.shape

    # nw to liczba dokumentów (kolumn), w których termin w (wiersz) występuje
    # Dla csr_matrix, liczymy niezerowe elementy w każdym wierszu
    nw = np.array(matrix.astype(bool).sum(axis=1)).ravel()

    idf_vector = np.log(N / nw)
    return idf_vector

def TF_IDF_calc(matrix):
    save_dir = "."
    # Tworzymy macierz diagonalną z wektora IDF.
    # Ta macierz będzie miała wartości IDF na głównej diagonalnej.
    # Wymiar macierzy diagonalnej: (num_terms, num_terms)
    idf_vector = IDF_calc(matrix)
    idf_diag_matrix = diags(idf_vector, format="csr")  #macierz rzadka

    # Mnożenie macierzowe: IDF_diag * matrix
    # Wynikowa macierz TF-IDF będzie miała te same wymiary i strukturę rzadkości co TF_matrix (matrix).
    # (m x m) @ (m x n) -> (m x n)
    tfidf_matrix = idf_diag_matrix @ matrix

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "idf_vector.npy"), idf_vector)
    print("Zapisano IDF_VECTOR!")
    np.save(os.path.join(save_dir, "tfidf_matrix.npy"), tfidf_matrix)
    print("Zapisano TF_IDF_MATRIX!")
    return tfidf_matrix

vocab_map = load_big_bow("big_bow.json")
docs_data_with_names = load_processed_documents_from_files("docs")
term_document_matrix, doc_names_in_matrix_order = create_matrix(docs_data_with_names, vocab_map)

matrix = TF_IDF_calc(term_document_matrix)

from sklearn.decomposition import TruncatedSVD
def get_ready_for_svd(matrix, k):
    # TruncatedSVD oczekuje macierzy (Dokumenty, Termy)
    # Nasza macierz jest (Termy, Dokumenty), więc transponujemy ją
    matrix = matrix.T
    svd_model = TruncatedSVD(n_components=k)
    svd_model.fit(matrix)
    us_matrix = svd_model.transform(matrix)
    v_t_matrix = np.array(svd_model.components_)
    with open("us.npz", 'wb') as file:
        np.save(file, us_matrix)
    with open("v_t.npz", 'wb') as file:
        np.save(file, v_t_matrix)
    print("Zapisano v_t i us!")

get_ready_for_svd(matrix,k=100)