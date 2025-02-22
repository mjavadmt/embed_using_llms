import os
import openai
from dotenv import load_dotenv
import numpy as np

from src.open_ai import get_code_embedding

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def compare_code_similarity(code1, code2):
    embedding_intentionally_long_identifier = get_code_embedding(code1)
    embedding2 = get_code_embedding(code2)

    # to be used instead of embedding1 - > embedding_intentionally_long_identifier
    if embedding_intentionally_long_identifier is None or embedding2 is None:
        return None

    # Calculate cosine similarity
    similarity = np.dot(embedding_intentionally_long_identifier, embedding2) / (
            np.linalg.norm(embedding_intentionally_long_identifier) * np.linalg.norm(embedding2)
    )

    return similarity


if __name__ == "__main__":
    python_code = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """

    similar_code = """
    def fib(n):
        if n <= 1:
            return n
        return fib(n-1) + fib(n-2)
    """

    different_code = """
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    """

    embedding = get_code_embedding(python_code)
    if embedding:
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First few values: {embedding[:5]}")

    similarity1 = compare_code_similarity(python_code, similar_code)
    similarity2 = compare_code_similarity(python_code, different_code)

    if similarity1 and similarity2:
        print(f"Similarity with similar code: {similarity1:.4f}")
        print(f"Similarity with different code: {similarity2:.4f}")