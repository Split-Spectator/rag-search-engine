from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords, CACHE_DIR
from nltk.stem import PorterStemmer
import string
import os
import pickle
from collections import defaultdict

stemmer = PorterStemmer()
movies = load_movies()
stopWords = load_stopwords()
class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id:int, text:str) -> None:
        tokens = tokenize_text(text=text)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def get_documents(self, term:str)->list[int]:
        ids = sorted(list(self.index.get(term, set())))
        return ids

    def build(self)->None:
        for movie in movies:
            doc_id = movie['id']
            doc_description = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, doc_description)
            self.docmap[doc_id] = movie
    def save(self)->None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, 'wb') as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, 'wb') as file:
            pickle.dump(self.docmap, file)

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    results = []
    query_tokens = tokenize_text(query)
    title_tokens = tokenize_text(movie["title"])
    for movie in movies:   
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
        if len(results) >= limit:
            break
    return results

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    clean_tokens = [stemmer.stem(token) for token in tokens if token and token not in stopWords]
    return clean_tokens


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

 