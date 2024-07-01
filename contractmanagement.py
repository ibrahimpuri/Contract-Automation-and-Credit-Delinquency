import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from openai import OpenAI
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import json
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client.api_key:
    logger.error("OpenAI API key not found. Please set it in your .env file.")
    exit(1)

# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example documents (in a real scenario, these would be loaded from a database)
documents = [
    {"id": "1", "content": "The contract stipulates that delivery must be made within 30 days.",
     "meta": {"source": "contract1"}},
    {"id": "2", "content": "The payment terms are net 30 days.", "meta": {"source": "contract2"}},
    {"id": "3", "content": "The warranty period for the product is 12 months from the date of purchase.",
     "meta": {"source": "contract3"}},
    {"id": "4", "content": "Any disputes arising from this contract shall be resolved through arbitration.",
     "meta": {"source": "contract4"}},
    {"id": "5", "content": "The client agrees to pay a penalty of 2% for late payments.",
     "meta": {"source": "contract5"}},
]

# Precompute embeddings for all documents
document_embeddings = [model.encode(doc['content']) for doc in documents]


def encode_text(text: str) -> np.ndarray:
    """Encode text using the sentence transformer model."""
    return model.encode(text)


def semantic_search(query: str, top_k: int = 3) -> List[Dict]:
    """Perform semantic search on the documents."""
    query_embedding = encode_text(query)
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [documents[i] for i in top_indices]


def query_gpt3_turbo(prompt: str, max_tokens: int = 150) -> str:
    """Query GPT-3.5 Turbo with a given prompt."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant specialized in contract analysis. Provide concise and accurate information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error querying GPT-3.5 Turbo: {str(e)}")
        raise


def contract_automation(query: str) -> Dict:
    """Perform contract automation including retrieval and analysis."""
    try:
        # Retrieve documents based on the query
        retrieved_docs = semantic_search(query)

        # Extract relevant content from the retrieved documents
        relevant_texts = " ".join([doc['content'] for doc in retrieved_docs])

        # Generate a summary or response using GPT-3.5 Turbo
        prompt = f"""Based on the following contract text, provide a detailed analysis answering the question: {query}

Relevant contract text:
{relevant_texts}

Provide your analysis in JSON format with the following structure:
{{
    "summary": "A brief summary of the answer",
    "key_points": ["List", "of", "key", "points"],
    "confidence": "A value between 0 and 1 indicating your confidence in the answer"
}}"""

        gpt_response = query_gpt3_turbo(prompt, max_tokens=300)

        # Parse the JSON response
        try:
            parsed_response = json.loads(gpt_response)
        except json.JSONDecodeError:
            logger.warning("GPT-3.5 Turbo response was not valid JSON. Falling back to raw text.")
            parsed_response = {"summary": gpt_response, "key_points": [], "confidence": 0.5}

        # Prepare the result
        result = {
            "query": query,
            "analysis": parsed_response,
            "sources": [{"id": doc["id"], "content": doc["content"]} for doc in retrieved_docs],
            "timestamp": datetime.now().isoformat()
        }

        return result
    except Exception as e:
        logger.error(f"Error in contract automation: {str(e)}")
        raise


def main():
    print("Welcome to the Contract Analysis Tool")
    print("-------------------------------------")
    while True:
        query = input("\nEnter your contract query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            print("Thank you for using the Contract Analysis Tool. Goodbye!")
            break
        try:
            result = contract_automation(query)
            print("\nAnalysis Result:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again with a different query.")


if __name__ == "__main__":
    main()