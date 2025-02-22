import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_code_embedding(code_snippet):
    """
    Get embeddings for code snippets using OpenAI's API.
    """
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-large",
            input=code_snippet,
            encoding_format="float",
            dimensions=3072  # Optional: maximum dimensions for more detailed code representation
        )

        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


if __name__ == "__main__":
    sample_code = "def hello_world():\n    print('Namaste!')"

    embedding = get_code_embedding(sample_code)
    if embedding:
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First few values: {embedding[:5]}")

