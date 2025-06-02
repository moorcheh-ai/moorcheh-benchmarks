import os
import csv
import time
import pandas as pd
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Set up API keys and config
OPENAI_API_KEY = "Your_OPENAI_API_KEY"  # Replace this
QDRANT_URL = "Your_QDRANT_URL"     # Or your Qdrant Cloud endpoint
QDRANT_API_KEY = "Your_QDRANT_API_KEY" # Replace this
COLLECTION_NAME = "Your_Collection_Name" # Replace this

client = OpenAI(api_key=OPENAI_API_KEY)

# Load PDF and split into chunks
pdf_path = "pdf_path"  # Replace this
query_csv_path = "queries.csv" 
output_csv_path = "results/results.csv"

loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  # all-MiniLM-L6-v2 output size

# Initialize Qdrant
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Create collection if not exists
if COLLECTION_NAME not in qdrant.get_collections().collections:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )

# Upload documents
points = []
for i, chunk in enumerate(chunks):
    text = chunk.page_content.strip()
    vector = embedder.encode(text).tolist()
    points.append(PointStruct(id=i, vector=vector, payload={"text": text}))

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

# Retrieval function
def retrieve_context(query, k=5):
    query_vector = embedder.encode(query).tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k
    )
    return [hit.payload["text"] for hit in results] or ["[No context retrieved]"]

# Generate answer
def generate_answer(query):
    context = retrieve_context(query)
    prompt = f"""Answer the following question using the provided context.

Context:
{chr(10).join(context)}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content, context

# --- Read queries and generate answers ---
queries_df = pd.read_csv(query_csv_path)
os.makedirs("results", exist_ok=True)

counter = 0
with open(output_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["passage_id", "query", "generated_answer", "passage"])
    writer.writeheader()

    for q in queries_df["query"]:
        print(f"Processing: {q}")
        try:
            answer, context = generate_answer(q)
            writer.writerow({
                "passage_id": counter,
                "query": q,
                "generated_answer": answer,
                "passage": " ".join(context)
            })
            counter += 1
        except Exception as e:
            print(f"Error for query '{q}':", e)
