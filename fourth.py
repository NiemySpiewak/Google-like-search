import numpy as np
import os
import json
import time
from scipy.sparse import load_npz, diags, lil_matrix, csr_matrix
CORPUS_DIR_FOR_URL_RETRIEVAL = "docs"
#czy możemy używać biblioteki do normalizacji?
from sklearn.preprocessing import normalize

def BagOfWords(text):
    list_of_words = text.split()
    print(list_of_words)
    bag = {}
    for word in list_of_words:
        if word in bag:
            bag[word] +=1
        else:
            bag[word] = 1
    return bag
def BOW_to_vector(small_bow,big_bag):
    matrix = lil_matrix((len(big_bag), 1), dtype=np.float64)
    for word,count in small_bow.items():
            if word in big_bag:
                idx = big_bag[word]
                matrix[idx] = count
    return matrix.tocsr()
def load_big_bow(path):
    vocab_map = {}

    with open(path, "r", encoding="utf-8") as f:
        vocab_map = json.load(f)

    return vocab_map
def load_tfidf_and_idf_vector(directory=".") -> tuple[csr_matrix | None, np.ndarray | None]:
    idf_vector = np.load("idf_vector.npy")
    loaded_array_container = np.load("tfidf_matrix.npy", allow_pickle=True)
    actual_sparse_matrix_object = loaded_array_container.item()
    if isinstance(actual_sparse_matrix_object, csr_matrix):
        tfidf_csr_matrix = actual_sparse_matrix_object

    return tfidf_csr_matrix, idf_vector
def load_doc_order(path):
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            doc_order = json.load(f)
        return doc_order
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Błąd podczas wczytywania pliku '{path}': {e}")
        return []

def get_url_from_document_file(filename: str, corpus_directory: str = CORPUS_DIR_FOR_URL_RETRIEVAL) -> str | None:
    """Odczytuje pierwszą linię pliku i próbuje wyekstrahować z niej URL."""
    filepath = os.path.join(corpus_directory, filename)
    if not os.path.exists(filepath):
        print(f"Ostrzeżenie: Plik '{filepath}' nie znaleziony przy próbie odczytu URL.")
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line.lower().startswith("url:"):
                return first_line[4:].strip() # Zwróć tekst po "URL: "
            else:
                # Jeśli pierwsza linia nie zaczyna się od "URL:", możemy założyć, że to sam URL
                # lub zwrócić None/first_line w zależności od oczekiwań
                # print(f"Ostrzeżenie: Pierwsza linia pliku '{filename}' nie zaczyna się od 'URL:'. Zwracam całą linię.")
                return first_line # Lub None, jeśli URL musi mieć prefix
    except Exception as e:
        print(f"Błąd podczas odczytywania URL z pliku '{filepath}': {e}")
    return None

def wyszukiwarka(InputText,doc_order,big_bow, matrix,idf_vector, use_svd, max_doc = 10):
    start_time = time.time()
    #InputText -> BagOfWords
    bow_for_input_text = BagOfWords(InputText)
    #BagOfWords -> Wektor
    vector = BOW_to_vector(bow_for_input_text, big_bow)

    # Tworzymy rzadką macierz diagonalną z wektora IDF
    # To jest efektywny sposób na przemnożenie każdego "wiersza" (elementu)
    # query_tf_vector przez odpowiadający mu IDF.
    idf_diag_q = diags(idf_vector, format="csr") # (m x m)
    
    # Mnożenie macierzowe: IDF_diag * TF_vector_query
    # (m x m) @ (m x 1) -> (m x 1)
    tfidf_vector = idf_diag_q @ vector

    #Normalizacja wektora zapytania
    normalized_tfidf_vector = normalize(tfidf_vector, norm='l2', axis=0)

    # Normalizacja kolumn macierzy dokumentów
    normalized_matrix = normalize(matrix, norm='l2', axis=0)
    
    #prawdopodobieństwo cosinusowe
    #probability_vector będzie teraz wektorem NumPy 1D o długości n (liczba dokumentów), 
    #gdzie probability_vector[j] to podobieństwo kosinusowe między zapytaniem a j-tym dokumentem.
    if use_svd:
        # Używamy SVD do redukcji wymiarów
        print("Użycie TruncatedSVD do redukcji wymiarowości...")
        PRELOADED_VT_MATRIX = np.load("v_t.npz")
        PRELOADED_US_MATRIX = np.load("us.npz")
        query_lsi_col_vector = PRELOADED_VT_MATRIX @ tfidf_vector
        normalized_query_lsi = normalize(query_lsi_col_vector, norm='l2', axis=0)
        normalized_docs_lsi = normalize(PRELOADED_US_MATRIX, norm='l2', axis=1)
        probability_vector = (normalized_docs_lsi @ normalized_query_lsi).ravel()
        print(f"DEBUG: Długość probability_vector: {len(probability_vector)}")
        print(f"DEBUG: Długość doc_order_global_list: {len(doc_order)}")

    else:
        # Zwykłe podobieństwo kosinusowe bez SVD
        probability_vector = (normalized_tfidf_vector.T @ normalized_matrix).toarray().ravel()

    # Sortowanie wyników
    doc_lista = list(enumerate(probability_vector))
    sorted_doc_lista = sorted(doc_lista, key=lambda x: x[1], reverse=True)

    result = []
    print(f"\nTop {max_doc} najbardziej podobnych dokumentów do zapytania '{InputText}':")
    for i in range(min(max_doc, len(sorted_doc_lista))):
        doc_matrix_index = sorted_doc_lista[i][0]
        similarity_score = sorted_doc_lista[i][1]

        if similarity_score > 1e-6: # Użyj spójnej nazwy argumentu
            if doc_matrix_index < len(doc_order):
                document_filename = doc_order[doc_matrix_index] # Nazwa pliku
                
                # Odczytaj URL z pliku
                document_url = get_url_from_document_file(document_filename)
                
                display_identifier = document_url if document_url else document_filename # Użyj URL jeśli dostępny

                result.append({"document_identifier": display_identifier, "similarity": similarity_score})
                print(f"  {i+1}. Dokument: {display_identifier}, Podobieństwo kosinusowe: {similarity_score:.4f}")
            else:
                print(f"  {i+1}. Błąd: Indeks dokumentu {doc_matrix_index} poza zakresem listy nazw dokumentów ({len(doc_order)}).")
        else:
            if i == 0:
                print("Nie znaleziono dokumentów o znaczącym podobieństwie.")
            break
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nCzas wyszukiwania: {elapsed_time:.3f} sekundy")
    return result

def do(InputText, flag):
    vocab_map = load_big_bow("big_bow.json")
    if vocab_map:
        print("vocab_map OK")
    matrix, vector = load_tfidf_and_idf_vector()
    if matrix is not None and vector is not None:
        print("matrix and vector OK")
    doc_order = load_doc_order("document_order.json")
    if doc_order is not None:
        print("doc_order OK")
    print("ready")
    result = wyszukiwarka(
        InputText, 
        doc_order=doc_order,  
        big_bow = vocab_map,                                  
        matrix=matrix,                                          
        idf_vector=vector,                                      
        use_svd=flag,                                           
        max_doc=5                                               
    )
    return result