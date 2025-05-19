import string
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
import re
import os

first_urls = ["https://en.wikipedia.org/wiki/Artificial_intelligence"]
max_doc = 20000
stop_words = [
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'also',
    'it', 'its', 'of', 'for', 'with', 'in', 'on', 'to', 'and', 'or', 'if', 'this',
    'that', 'then', 'not'
]

def text_cleaning(text,stop_words):

    text = re.sub(r'\[.*?\]', '', text) # Usuwa wszelkie nawiasy kwadratowe z zawartością, np. [1], [note]
    text = text.strip() # Usuwa spacje z początku i końca tekstu
    text = text.lower()
    # Zastąpienie wszystkich znaków interpunkcyjnych spacjami
    # string.punctuation = !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    punctuation_to_space_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(punctuation_to_space_table)

    text = re.sub(r'\s+', ' ', text)
    text = remove_stop_words(text,stop_words)
    return text

def remove_stop_words(text, stop_words):
    words = text.split()
    result = []
    for word in words:
        if word not in stop_words:
            result.append(word)
    final_string = " ".join(result)
    return final_string

def check_url(url,parsed_url):
    DOMAIN = "en.wikipedia.org"
    if (parsed_url.scheme in ['http','https'] and
        parsed_url.netloc == DOMAIN and
        parsed_url.path.startswith('/wiki/') and
        ':' not in parsed_url.path and
        'Main_Page' not in parsed_url.path and
        'disambiguation' not in parsed_url.path.lower() and
        '#' not in url):
        return True
    else:
        return False

def crawler(first_urls,stop_words,max_doc):
    stop_words = set(stop_words)

    urls_queue = list(first_urls)
    visited_urls = set(first_urls)
    counter = 0

    print("Start")

    while urls_queue and counter < max_doc:
        url = urls_queue.pop(0)
        print(f"Przetwarzam: {url}, counter: {counter}")

        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')


        #wyciągmy to co chcemy z tekstu

        #1. usuwamy niechciane rzeczy aby móc zrobić find
        for unwanted_tag in soup(["nav", "footer", "script", "style", ".mw-editsection", ".toc", ".reference", ".external.text", ".noprint"]):
            unwanted_tag.decompose()

        #2. Celowanie w główny kontener treści
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if content_div:
            parser_output_div = content_div.find('div', class_='mw-parser-output')
            if parser_output_div:
                #3. Ekstrakcja tekstu z paragrafów w tym kontenerze
                paragraphs = parser_output_div.find_all('p', recursive=True)
                text_content = "\n".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
            else:
                text_content = content_div.get_text(separator=" ", strip=True)
        else:
            print(f"   Nie znaleziono 'mw-content-text' dla {url}. Używam body.")
            body = soup.find('body')
            if body:
                text_content = body.get_text(separator=" ", strip=True)
            else:
                text_content = ""

        # Proste czyszczenie tekstu
        text_content = text_cleaning(text_content,stop_words)
        

        # Zapisz tylko jeśli jest wystarczająco dużo tekstu
        if len(text_content) > 200:
            doc_filename = os.path.join("docs", f"doc_{counter:04d}.txt")
            with open(doc_filename, "w", encoding="utf-8") as f:
                f.write(f"URL: {url}\n\n") # Zapisujemy URL jako pierwszą linię dla referencji
                f.write(text_content)
            
            print(f"   Zapisano: {doc_filename}")
            counter += 1
        else:
            print(f"   Pominięto (za mało treści): {url}")


        #szukanie linków
        if counter < max_doc and len(urls_queue)< max_doc:
            for link in soup.find_all('a',href = True):
                new_link = link['href']
                new_url = urljoin(url,new_link)
                parsed_url = urlparse(new_url)

                flag = check_url(new_url,parsed_url)
                if new_url not in visited_urls and flag == True:
                    if len(urls_queue)< max_doc:
                        urls_queue.append(new_url)
                        visited_urls.add(new_url)

crawler(first_urls,stop_words,max_doc)