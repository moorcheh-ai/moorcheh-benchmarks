# Build your own powerful RAG pipeline!
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system with Elastic Search.

# --- Installation ---
# First, make sure you have these essential libraries installed.
# You can run this command in your terminal:
# pip install elasticsearch openai sentence-transformers langchain-community pypdf pandas google-generativeai

# --- Import Libraries ---
import os
import csv
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import google.generativeai as genai   # Optional, for future integration with Gemini

# --- Configuration ---
# Set the API keys securely using environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)  # GPT client setup

# Define file paths
PDF_PATH = "your-pdf-path.pdf"  # Replace with your PDF path
QUERY_CSV = "your-query-file.csv"
OUTPUT_CSV = "results/results.csv"
INDEX_NAME = "rag_docs"
TOP_K = 5

# --- Connect to Elasticsearch ---
# Be sure to set ELASTIC_URL and ELASTIC_API_KEY in your environment
es = Elasticsearch(
    os.environ["ELASTIC_URL"],
    api_key=os.environ["ELASTIC_API_KEY"]
)

# --- Initialize Embedder ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Load sentence transformer model

# --- Load and Chunk PDF ---
loader = PyPDFLoader(PDF_PATH)  # Load PDF file
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)  # Setup chunker
chunks = splitter.split_documents(pages)

# --- Initialize Elastic Search Index ---
es.indices.create(index=INDEX_NAME, body={
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "metadata": {"type": "object"},
            "vector": {"type": "dense_vector", "dims": 384}
        }
    }
})

# --- Upload Chunks with Embeddings to Elasticsearch ---
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
bulk(es, docs)  # Bulk insert into Elasticsearch

# --- Retrieval Function ---
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

# --- Generation Function ---

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
    
    # --- For GPT integration ---
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content, context
  

# --- Read Queries and Generate Answers ---
queries_df = pd.read_csv(QUERY_CSV)
os.makedirs("results", exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["query", "generated_answer", "passage"])
    writer.writeheader()

    for q in queries_df["query"]: # Iterate through each query
        print(f"Processing: {q}")
        try:
            answer, context = generate_answer(q) # Generate answer using RAG
            writer.writerow({
                "query": q,
                "generated_answer": answer,
                "passage": "\n\n====================\n\n".join(context) # Separate passages for clarity
            })
        except Exception as e:
            print(f"Error for query '{q}':", e)
