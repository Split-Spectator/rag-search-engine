from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords, CACHE_DIR
from collections  import defaultdict, Counter
from nltk.stem import PorterStemmer
import string
import os
import pickle
import math
from collections import defaultdict

stemmer = PorterStemmer()
movies = load_movies()
stopWords = load_stopwords()
class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.term_frequencies = defaultdict(Counter)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id:int, text:str) -> None:
        tokens = tokenize_text(text=text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)  #here 

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
        with open(self.tf_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.tf_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:        
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("term must be a single token")
        if len(tokens) == 0:
            return 0
        token = tokens[0]
        return self.term_frequencies[doc_id][token]
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        term_doc_count = len(self.index[token])
        doc_count = len(self.docmap)
        return math.log((doc_count + 1)  / (term_doc_count + 1))

    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1) 

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    return idx

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    results = []
    query_tokens = tokenize_text(query)
    idx = InvertedIndex()
    idx.load()
    seen = set()
    for q in query_tokens:
        ids = idx.get_documents(q)
        for id in ids:
            if id in seen:
                continue
            seen.add(id)
            results.append(idx.docmap[id])
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


def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(term)

def tfidf_command(doc_id: int, terms: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, terms)

def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

