from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords
import string

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:

    movies = load_movies()
    stopWords = load_stopwords()
    results = []
    for movie in movies:   
        query_tokens = tokenize_text(query, stopWords)
        title_tokens = tokenize_text(movie["title"], stopWords)
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
        if len(results) >= limit:
            break
    return results

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str, stopWords: list[str]) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    clean_tokens = [token for token in tokens if token and token not in stopWords]
    return clean_tokens


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False