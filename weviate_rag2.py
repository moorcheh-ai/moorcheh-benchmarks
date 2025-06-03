# Build your own RAG pipeline! 
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system with Weviate.

#--- Installation ---
# First, make sure you have these essential libraries installed.
# You can run this command in your terminal:
# pip install openai weaviate-client langchain_community pypdf pandas

#--- Import Libraries ---

import os
import csv
import time
import pandas as pd
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.auth import AuthApiKey
import uuid

# --- Configuration ---
OPENAI_API_KEY = "Your_OPENAI_API_KEY"  # Replace this
WEAVIATE_URL = "Your_URL"  # Replace with your WCS instance URL
WEAVIATE_API_KEY = "Your_WEVIAtE_API_KEY"  # Replace this
COLLECTION_NAME = "Your_COLLLECTION_NAME"  # Replace this 

pdf_path = "pdf-path"  # Replace with your actual PDF path
query_csv_path = "queries.csv"
output_csv_path = "results/results.csv"

# --- Initialize OpenAI Client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Load and Split PDF ---
loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

# --- Embedding Model ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  # embedding dimension for all-MiniLM-L6-v2

# --- Connect to Weaviate ---
auth = AuthApiKey(WEAVIATE_API_KEY)
client_wv = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=auth,
)

# --- Create collection if it doesn't exist ---
if not client_wv.collections.exists(COLLECTION_NAME):
    client_wv.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none(),
        vector_index_config=weaviate.classes.config.Configure.VectorIndex.hnsw(
            distance_metric=weaviate.classes.config.VectorDistances.COSINE
        ),
        properties=[
            weaviate.classes.config.Property(name="text", data_type=weaviate.classes.config.DataType.TEXT)
        ]
    )

collection = client_wv.collections.get(COLLECTION_NAME)

# --- Upload Chunks to Weaviate ---
for i, chunk in enumerate(chunks):
    text = chunk.page_content.strip()
    vector = embedder.encode(text).tolist()
    collection.data.insert(
        uuid=str(uuid.uuid4()),
        properties={"text": text},
        vector=vector
    )

# --- Retrieve context from Weaviate ---
def retrieve_context(query, k=5):
    query_vector = embedder.encode(query).tolist()
    result = collection.query.near_vector(query_vector, limit=k)
    return [obj.properties["text"] for obj in result.objects] or ["[No context retrieved]"]

# --- Generate Answer from OpenAI ---
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

# --- Read queries and write answers ---
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
                "passage": " ".join(context)
            })
        except Exception as e:
            print(f"Error for query '{q}':", e)

client_wv.close()