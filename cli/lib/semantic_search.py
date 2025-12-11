from sentence_transformers import SentenceTransformer
from .search_utils import CACHE_DIR, load_movies
import numpy as np
import os

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

def embed_text(text):
    semINT = SemanticSearch()
    embedding = semINT.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
    
    def generate_embedding(self, text):
        search_instance = SemanticSearch()
        if not text.strip():
            raise ValueError("Input string is empty or consists only of whitespace characters.")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents): 
        self.documents = documents
        result = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            result.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(result, show_progress_bar=True)

        os.makedirs(os.path.dirname(CACHE_DIR), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)      
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)


def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")



def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {search_instance.embeddings.shape[0]} vectors in {search_instance.embeddings.shape[1]} dimensions")

def embed_query_text(query):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


