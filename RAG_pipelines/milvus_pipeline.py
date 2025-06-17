# Build your own RAG pipeline! 
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system with Milvus.

#--- Installation ---
# First, make sure you have these essential libraries installed.
# You can run this command in your terminal:
# pip install openai weaviate-client langchain_community pypdf pandas google-generativeai

#--- Import Libraries ---
import os
import csv
import time
import pandas as pd
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import connections, MilvusClient, FieldSchema, CollectionSchema, DataType, Collection
import google.generativeai as genai   # Optional, for future integration with Gemini

# --- Configuration ---
OPENAI_API_KEY = "your-API-key"
MILVUS_COLLECTION_NAME = "rag_docs"
pdf_path = "pdf-path"
query_csv_path = "queries.csv"
output_csv_path = "results/results.csv"

# --- Connect to Milvus ---
connections.connect(alias="default", host="localhost", port="19530")
client = MilvusClient(uri="http://localhost:19530")

# --- SentenceTransformer embedder ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
DIM = 384  # embedding dimension

# --- Drop existing collection if exists ---
if client.has_collection(MILVUS_COLLECTION_NAME):
    client.drop_collection(MILVUS_COLLECTION_NAME)

# --- Define schema ---
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM)
]
schema = CollectionSchema(fields=fields, description="RAG PDF chunks")

client.create_collection(
    collection_name=MILVUS_COLLECTION_NAME,
    schema=schema,
    consistency_level="Strong"
)

# --- Load PDF and split into chunks ---
loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
chunks = splitter.split_documents(pages)

# --- Prepare insertion data as a list of dictionaries ---
data_to_insert = []
for i, chunk in enumerate(chunks):
    text = chunk.page_content.strip()
    vector = embedder.encode(text)
    vector = [float(x) for x in vector]  # Ensure Python floats
    data_to_insert.append({
        "id": i,
        "text": text,
        "embedding": vector
    })

# --- Insert data into Milvus ---
client.insert(
    collection_name=MILVUS_COLLECTION_NAME,
    data=data_to_insert
)

# --- Create Index ---
collection = Collection(name=MILVUS_COLLECTION_NAME)

collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

collection.load()

# --- Retrieval function ---
def retrieve_context(query, k=5):
    query_emb = embedder.encode(query)
    query_emb = [float(x) for x in query_emb]
    results = client.search(
        collection_name=MILVUS_COLLECTION_NAME,
        data=[query_emb],
        anns_field="embedding",
        search_params={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=k,
        output_fields=["text"]
    )
    return [hit.entity.get("text") + " [] " for hit in results[0]] if results[0] else ["[No context retrieved]"]

# --- OpenAI client for answer generation ---
client = OpenAI(api_key=OPENAI_API_KEY)

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
50 = The context is partially related (e.g. correct company but wrong financial quarter)
0 = The context is topically unrelated to the query

Rationale:
Briefly explain what aspects of the context are topically aligned with the query. If the context includes off-topic information, describe it.

2. Completeness Evaluation
If someone were to answer the query using only this context, how complete and sufficient would their answer be?

Provide a completeness score between 0 and 100, where:

100 = The context includes all necessary information to fully answer the query
50 = The context includes some, but not all, key information (e.g., only Q1 revenue when the query asks for full-year revenue)
0 = The context includes none of the necessary information to answer the query

Rationale:
Clearly state whether the context contains the required facts, figures, or explanations needed to construct a complete answer. If any crucial components are missing, specify what they are.

Context:
{chr(10).join(context)}

Question: {query}
Answer:"""

    # --- For Gemini integration ---
    # response = model.generate_content(prompt)
    # return response.text, context

    # --- For GPT integration ---
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

# --- Run evaluation loop ---
queries_df = pd.read_csv(query_csv_path)
os.makedirs("results", exist_ok=True)

with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["passage_id", "query", "generated_answer", "passage"])
    writer.writeheader()
    counter = 0
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