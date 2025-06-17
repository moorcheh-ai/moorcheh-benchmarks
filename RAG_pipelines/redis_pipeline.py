# Build your own RAG pipeline! 
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system with Redis.

#--- Installation ---
# First, make sure you have these essential libraries installed.
# You can run this command in your terminal:
# pip install openai redis redisvl langchain_community pypdf pandas google-generativeai

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
import google.generativeai as genai   # Optional, for future integration with Gemini
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

# --- Delete old index if exists ---
try:
    r.ft(INDEX_NAME).dropindex(delete_documents=True)
except:
    pass

# --- Create index ---
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
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
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

#For Gemini integration:
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# model = genai.GenerativeModel("gemini-2.5-pro-preview-06-05")

def generate_answer(query):
    context = retrieve_context(query)
    prompt = f"""Answer the two following questions based on the retrieved passages provided from each query.

1. Relevance Evaluation  
Does the retrieved context directly pertain to the topic and scope of the query?

Provide a relevance score between 0 and 100, where:
100 = The context is entirely focused on the query’s subject  
50 = The context is partially related (e.g., correct company but wrong financial quarter)  
0 = The context is topically unrelated to the query

Rationale:  
Briefly explain what aspects of the context are topically aligned with the query. If the context includes off-topic information, describe it.

2. Completeness Evaluation  
If someone were to answer the query using only this context, how complete and sufficient would their answer be?

Provide a completeness score between 0 and 100, where:
100 = The context includes all necessary information to fully answer the query  
50 = The context includes some, but not all, key information  
0 = The context includes none of the necessary information

Rationale:  
Clearly state whether the context contains the required facts, figures, or explanations needed to construct a complete answer. If any crucial components are missing, specify what they are.

Context:
{chr(10).join(context)}

Question: {query}
Answer:"""
    
    # --- For Gemini integration ---
    # response = model.generate_content(prompt)
    # return response.text, context

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
                "passage": "\n\n====================\n\n".join(context)
            })
            counter += 1
        except Exception as e:
            print(f"Error for query '{q}':", e)