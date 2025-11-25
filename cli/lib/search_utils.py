import json
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
DEFAULT_SEARCH_LIMIT = 5



def load_movies() -> list[dict]:

    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOP_WORDS_PATH, "r") as f:
        stopWords = f.read().splitlines()
    return stopWords