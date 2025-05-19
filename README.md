# Semantic Search Engine with LSI (Latent Semantic Indexing)

This project implements a semantic search engine that leverages Latent Semantic Indexing (LSI) to find documents most relevant to a user's query. It processes a corpus of text documents, builds TF-IDF and LSI models, and provides a simple web interface for searching.

## Project Overview

The system operates in several key stages:

1.  **Data Collection (Crawler - `first.py`):**
    *   A web crawler (initiated from `first.py`) fetches English text documents, initially seeded from Wikipedia articles (e.g., "Artificial Intelligence").
    *   It extracts a predefined maximum number of documents (`max_doc` in `first.py`).
    *   The raw text content and its source URL are saved for each document.

2.  **Text Preprocessing (Embedded in `first.py` and `second.py`):**
    *   **Cleaning:** Removal of HTML-like square bracket annotations (e.g., `[1]`, `[edit]`).
    *   **Normalization:** Conversion to lowercase, replacement of punctuation with spaces, and normalization of whitespace.
    *   **Stopword Removal:** Common, low-information English stopwords are removed.
    *   The processed text for each document is saved, typically with the URL as the first line, followed by an empty line, then the cleaned text.

3.  **Vocabulary and Indexing (Corpus Analysis - `second.py`):**
    *   **Global Vocabulary Construction:** Iterates through all processed documents (from the `docs/` directory) to build a set of all unique terms (words). This vocabulary is saved to `vocabulary.txt`.
    *   **Word-to-Index Mapping:** Creates a JSON mapping (`big_bow.json`) where each unique term from the vocabulary is assigned a unique integer index.

4.  **Vector Space Model Construction (TF-IDF - `third.py`):**
    *   **Term Frequency (TF) Matrix:** Builds a sparse Term-Document matrix where rows represent terms (from `big_bow.json`) and columns represent documents. The values are the raw term frequencies (how many times a term appears in a document). The order of documents (columns) is saved in `document_order.json`.
    *   **Inverse Document Frequency (IDF) Calculation:** Computes the IDF score for each term in the vocabulary using the formula: `IDF(w) = log(N / nw)`. The resulting IDF vector is saved to `idf_vector.npy`.
    *   **TF-IDF Matrix:** Multiplies the TF matrix by the IDF scores to produce the final TF-IDF Term-Document matrix. This weighted matrix is saved as `tfidf_matrix.npy` (note: your code saves it as an `.npy` file containing a pickled SciPy sparse matrix object, which is loaded with `allow_pickle=True`).

5.  **Latent Semantic Indexing (SVD - `third.py`'s `get_ready_for_svd`):**
    *   **Singular Value Decomposition (SVD):** Applies TruncatedSVD to the *transposed* TF-IDF matrix (Document-Term orientation) to reduce dimensionality and capture latent semantic concepts.
    *   **LSI Components:**
        *   `us.npy`: Stores the transformed document representations in the LSI space (`Uk_doc * Sk_diag`, shape: `n_documents x k_components`).
        *   `v_t.npy`: Stores the term components in the LSI space (`Vk^T_term`, shape: `k_components x n_terms`).
    *   These components are pre-calculated and saved for efficient use during search.

6.  **Search Functionality (Query Processing - `four.py`'s `wyszukiwarka`):**
    *   **Query Preprocessing:** The user's input query undergoes the same text cleaning, tokenization, and stopword removal as the corpus documents.
    *   **Query Vectorization:** The processed query is converted into a TF vector using the global word-to-index map, and then into a TF-IDF vector using the global IDF scores.
    *   **Similarity Calculation:**
        *   **TF-IDF Mode:** If LSI is not used, the TF-IDF query vector is normalized, and the TF-IDF document matrix columns are normalized. Cosine similarity is then computed between the normalized query vector and each normalized document vector.
        *   **LSI Mode:** If LSI is used, the TF-IDF query vector is projected into the LSI space using the pre-calculated `v_t.npy` (term components). This projected query vector is then normalized. The pre-calculated `us.npy` (document LSI representations) are also normalized (if not already). Cosine similarity is computed in this lower-dimensional LSI space.
    *   **Ranking:** Documents are ranked by their cosine similarity scores in descending order.
    *   **Results:** The top `k` most similar documents (their original URLs) are returned.

7.  **User Interface (Front-end - `app_gradio.py`):**
    *   A simple web interface built with Gradio.
    *   Allows users to input a text query.
    *   Provides an option to toggle LSI (SVD) based search on/off (if LSI components are available).
    *   Allows users to specify the number of top results to display.
    *   Displays the search results, including document identifiers (URLs) and their similarity scores.


## Technologies and Libraries

*   **Python 3.x**
*   **NumPy:** For numerical operations.
*   **SciPy:** For sparse matrices (`csr_matrix`, `lil_matrix`, `diags`) and sparse SVD (though `sklearn.decomposition.TruncatedSVD` is used here).
*   **Scikit-learn:** For `TruncatedSVD` and vector normalization (`sklearn.preprocessing.normalize`).
*   **Requests:** For fetching web pages in the crawler.
*   **BeautifulSoup4:** For parsing HTML in the crawler.
*   **Gradio:** For creating the web-based user interface.
*   **JSON:** For storing and loading dictionaries and lists (vocabulary map, document order).
