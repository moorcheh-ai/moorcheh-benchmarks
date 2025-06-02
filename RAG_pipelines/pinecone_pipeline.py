import os
import csv
import pandas as pd
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import time

# Set up API keys and config
OPENAI_API_KEY = "Your-OpenAI-Key"  # Replace this
PINECONE_API_KEY = "Your-Pinecone-Key"  # Replace this
PINECONE_ENV = "us-east-1"  # Change this to your environment
INDEX_NAME = "rag-docs" # Change this to the name you want

client = OpenAI(api_key=OPENAI_API_KEY)

# Load PDF and split into chunks
pdf_path = "pdf-path"  # Replace this
query_csv_path = "queries.csv"
output_csv_path = "results/results.csv"

loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # all-MiniLM-L6-v2 has 384-dim vectors
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    # wait for index to be ready
    while pc.describe_index(INDEX_NAME).status['ready'] is False:
        time.sleep(1)

index = pc.Index(INDEX_NAME)

try:
    index.delete(delete_all=True)
except Exception as e:
    print("Index clear skipped:", e)

# Insert documents into Pinecone
for i, chunk in enumerate(chunks):
    text = chunk.page_content.strip()
    emb = embedder.encode(text).tolist()
    index.upsert(vectors=[{
        "id": str(i),
        "values": emb,
        "metadata": {"text": text}
    }])

# Retrieval function
def retrieve_context(query, k=5):
    query_emb = embedder.encode(query).tolist()
    results = index.query(vector=query_emb, top_k=k, include_metadata=True)
    return [match["metadata"]["text"] for match in results.get("matches", [])] or ["[No context retrieved]"]

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
