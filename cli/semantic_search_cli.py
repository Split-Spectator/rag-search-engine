#!/usr/bin/env python3
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text
import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings are loaded")
    
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a search query")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
   
    subparsers = subparsers.add_parser("embed_text", help="embed input text with model")
    subparsers.add_argument("text", type=str, help="text to get model embedding")

    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case _:
            parser.print_help()
        

if __name__ == "__main__":
    main()


def add_vectors(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("vector lengths dont match")
    result = [vec1[i] + vec2[i] for i in range(min(len(vec1), len(vec2)))]