# Build your own RAG pipeline! 
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system with Redis.

#--- Installation ---
# First, make sure you have these essential libraries installed.
# You can run this command in your terminal:
# pip install openai redis redisvl langchain_community pypdf pandas

#--- Import Libraries ---
import os
import csv
import time
import pandas as pd
import numpy as np
import redis
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
from redis.commands.search.field import TextField, VectorField
import redis

# --- CONFIG ---
OPENAI_API_KEY = "Your-OPENAI_API_KEY" # Replace this
REDIS_HOST = "YOUR_REDIS_URL"      # Or your Redis endpoint
REDIS_PORT = 16343 # or 6379 for non-SSL connection
INDEX_NAME = "Index_Name" # Replace this
VECTOR_DIM = 384
TOP_K = 5

pdf_path = "pdf-path"  # Replace this
query_csv_path = "queries.csv" 
output_csv_path = "results/results.csv"

client = OpenAI(api_key=OPENAI_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- REDIS SETUP ---
r = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT,
    password = 'Your_REDIS_DATABASE_PASSWORD', # Replase This
    decode_responses=False,
    ssl=False) 

# Delete old index if exists
try:
    r.ft(INDEX_NAME).dropindex(delete_documents=True)
except:
    pass

# Create index
r.ft(INDEX_NAME).create_index([
    redis.commands.search.field.TextField("text"),
    redis.commands.search.field.VectorField(
        "vector",
        "FLAT", {
            "TYPE": "FLOAT32",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": "COSINE",
            "INITIAL_CAP": 1000
        }
    )
])

# --- Load and Embed Documents ---
loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

for i, chunk in enumerate(chunks):
    text = chunk.page_content.strip()
    vector = embedder.encode(text).astype(np.float32).tobytes()
    r.hset(f"doc:{i}", mapping={"text": text, "vector": vector})

# --- Retrieval ---
def retrieve_context(query, k=TOP_K):
    query_vector = embedder.encode(query).astype(np.float32).tobytes()
    base_query = f"*=>[KNN {k} @vector $vec_param AS score]"
    q = Query(base_query).sort_by("score").return_fields("text", "score").dialect(2)
    results = r.ft(INDEX_NAME).search(q, query_params={"vec_param": query_vector})
    return [doc.text for doc in results.docs] or ["[No context retrieved]"]

# --- Answer Generation ---
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

# --- Process Queries ---
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