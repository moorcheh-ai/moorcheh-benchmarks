
# Build your own RAG pipeline! 
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system with MongoDB Atlas.

#--- Installation ---
# First, make sure you have these essential libraries installed.
# You can run this command in your terminal:
# pip install openai pymongo langchain_community pypdf pandas sentence-transformers google-generativeai
# Remember to set your vector index on MongoDB

#--- Import Libraries ---
import os
import csv
import time
import pandas as pd
import numpy as np
from openai import OpenAI
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- CONFIG ---
OPENAI_API_KEY = "Your_OPEN_AI_API_KEY" # Replace this
MONGO_URL = "mongodb+srv://<username>:<password>@<cluster-url>/" # Replace this with your cluster url (fill in the fields)
DB_NAME = "db_name" # Replace this
COLLECTION_NAME = "collection_name" # Replace this
VECTOR_DIM = 384
TOP_K = 5

pdf_path = "CombinedDoc.pdf"  # Replace this
query_csv_path = "queries.csv"
output_csv_path = "results/results.csv"

client = OpenAI(api_key=OPENAI_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- MongoDB Setup ---
mongo_client = MongoClient(MONGO_URL)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]


# --- Load and Embed Documents ---
loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
chunks = splitter.split_documents(pages)

# Insert documents with embeddings
docs_to_insert = []
for i, chunk in enumerate(chunks):
    text = chunk.page_content.strip()
    vector = embedder.encode(text).tolist()
    docs_to_insert.append({
        "doc_id": i,
        "text": text,
        "embedding": vector
    })
collection.insert_many(docs_to_insert)

# --- Retrieval (MongoDB Atlas Vector Search) ---
def retrieve_context(query, k=TOP_K):
    query_vector = embedder.encode(query).tolist()
    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": query_vector,
                "path": "embedding",
                "numCandidates": 100,
                "limit": k,
                "index": "vector_index"  # Replace with your Atlas Vector Search index name
            }
        },
        {
            "$project": {
                "text": 1,
                "_id": 0
            }
        }
    ]
    results = collection.aggregate(pipeline)
    return [doc["text"] for doc in results] or ["[No context retrieved]"]

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

with open(output_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["passage_id", "query", "generated_answer", "passage"])
    writer.writeheader()

    for counter, q in enumerate(queries_df["query"]):
        print(f"Processing: {q}")
        try:
            answer, context = generate_answer(q)
            writer.writerow({
                "passage_id": counter,
                "query": q,
                "generated_answer": answer,
                "passage": "\n\n====================\n\n".join(context)
            })
        except Exception as e:
            print(f"Error for query '{q}':", e)
