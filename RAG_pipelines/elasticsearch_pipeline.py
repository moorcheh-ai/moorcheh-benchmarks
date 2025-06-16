import os
import csv
import pandas as pd
from openai import OpenAI
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- Configuration ---
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # Your OpenAI key must be set in environment
ELASTIC_CLOUD_URL = os.environ["ELASTIC_URL"]  # Your Elastic Cloud endpoint 
ELASTIC_API_KEY = os.environ["ELASTIC_API_KEY"]  # Your Elastic Cloud API key

PDF_PATH = "path/to/your-pdf.pdf" # Path to your PDF document
QUERY_CSV = "queries.csv" # Path to your CSV file with queries
OUTPUT_CSV = "results/results.csv" # Where to save the results
INDEX_NAME = "rag_docs"  # Number of documents to retrieve
TOP_K = 5

# --- Initialize Clients ---
openai_client = OpenAI(api_key=OPENAI_API_KEY)
es = Elasticsearch(ELASTIC_CLOUD_URL, api_key=ELASTIC_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load and Chunk PDF ---
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120) # Split text into smaller pieces
chunks = splitter.split_documents(pages) # Divide PDF pages into chunks

# --- Create Index on Elastic-Search ---
es.indices.create(index=INDEX_NAME, body={
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "metadata": {"type": "object"},
            "vector": {"type": "dense_vector", "dims": 384}
        }
    }
})

# --- Upload PDF Chunks with Embeddings to Elasticsearch ---
docs = []
for i, chunk in enumerate(chunks):
    text = chunk.page_content.strip()
    embedding = embedder.encode(text).tolist()
    docs.append({
        "_index": INDEX_NAME,
        "_id": f"chunk_{i}",
        "_source": {
            "text": text,
            "metadata": {"chunk_index": i},
            "vector": embedding
        }
    })
bulk(es, docs)

# --- Retrieval Function using Cosine Similarity ---
def retrieve_context(query, k=TOP_K):
    query_vector = embedder.encode(query).tolist()
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    response = es.search(index=INDEX_NAME, size=k, query=script_query)
    hits = response["hits"]["hits"]
    return [hit["_source"]["text"] for hit in hits]

# --- Answer Generation using OpenAI GPT-4o ---
def generate_answer(query): # Function to create an answer from context
    context = retrieve_context(query)
    prompt = f"""Answer the question using the provided context.

Context:
{chr(10).join(context)}

Query: {query}
Answer:"""

    response = openai_client.chat.completions.create( # Ask OpenAI to generate an answe
        model="gpt-4o", # Using the GPT-4o model
        messages=[{"role": "user", "content": prompt}], # Your question for the AI
        temperature=0.2 # How creative the AI should be (lower is less creative)
    )
    return response.choices[0].message.content, context

# --- Run Evaluation for All Queries and Save Results ---
queries_df = pd.read_csv(QUERY_CSV)
os.makedirs("results", exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f: # Open the results CSV file
    writer = csv.DictWriter(f, fieldnames=["query", "generated_answer", "passage"]) # Load your questions from a CSV file
    writer.writeheader()

    for q in queries_df["query"]: # Go through each question
        print(f"Processing: {q}") # Show which question is being processed
        try:
            answer, context = generate_answer(q) # Get AI's answer and context
            writer.writerow({
                "query": q, # The original question
                "generated_answer": answer, # The AI-generated answer
                "passage": "".join(context) # All contexts used, joined into one string
            })
        except Exception as e: # If something goes wrong
            print(f"Error for query '{q}':", e) # Print the error
