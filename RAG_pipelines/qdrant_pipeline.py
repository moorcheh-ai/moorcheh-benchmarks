# Build your own powerful RAG pipeline!
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system with Qdrant.

# --- Installation ---
# First, make sure you have these essential libraries installed.
# You can run this command in your terminal:
# pip install qdrant_client openai sentence-transformers langchain-community pypdf pandas google-generativeai

# --- Import Libraries ---

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
import google.generativeai as genai   # Optional, for future integration with Gemini

# --- Configuration ---
# Set the API keys securely using environment variables
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
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
chunks = splitter.split_documents(pages)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  # all-MiniLM-L6-v2 output size

# --- Initialize Qdrant ---
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- Create collection if it does not exist ---
if COLLECTION_NAME not in qdrant.get_collections().collections:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )

# --- Upload documents ---
points = []
for i, chunk in enumerate(chunks):
    text = chunk.page_content.strip()
    vector = embedder.encode(text).tolist()
    points.append(PointStruct(id=i, vector=vector, payload={"text": text}))

qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

# --- Retrieval function ---
def retrieve_context(query, k=5):
    query_vector = embedder.encode(query).tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=k
    )
    return [hit.payload["text"] for hit in results] or ["[No context retrieved]"]

# --- Generate answer ---
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
