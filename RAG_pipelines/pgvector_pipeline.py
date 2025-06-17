# Build your own powerful RAG pipeline!
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system using pgvector.

# --- Installation ---
# First, make sure you have these essential libraries installed.
# You can run this command in your terminal:
# pip install openai langchain_community pypdf pandas psycopg2-binary python-dotenv sentence-transformers google-generativeai

# --- Import Libraries ---
import os
import csv
import json  
import psycopg2
import pandas as pd
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
#import google.generativeai as genai  # Optional Gemini integration

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
# Set your OpenAI API key
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] # Replace with your OpenAI API key
openai_client = OpenAI(api_key=OPENAI_API_KEY) # Initialize OpenAI client

# Connect to your local PostgreSQL database with pgvector installed
conn = psycopg2.connect(
    dbname="your-database-name", # Your Postgres database name
    user="your-username",     # Your database username
    password="your-password", # Your password (leave blank if not needed)
    host="localhost",         # Host for local DB
    port="5432"               # Default Postgres port
)
cursor = conn.cursor()

# --- File paths ---
pdf_path = "your-pdf-path.pdf"                   # Input PDF path
query_csv_path = "your-query-file.csv"           # CSV with input queries
output_csv_path = "results/results.csv"          # Output CSV path
queries_df = pd.read_csv(query_csv_path)

# --- Load and Chunk PDF ---
loader = PyPDFLoader(pdf_path)
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
chunks = splitter.split_documents(pages)

# --- Generate Embeddings and Upload Chunks to pgvector ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return embedder.encode(text).tolist()

# Loop through each chunk and insert it into the database
for i, chunk in enumerate(chunks):
    text = chunk.page_content.strip()
    embedding = get_embedding(text)

# Insert into your "table" (must have columns: chunk_index, content, embedding, metadata)
    cursor.execute(
        "INSERT INTO table (chunk_index, content, embedding, metadata) VALUES (%s, %s, %s, %s)",
        (i, text, embedding, json.dumps({"source": "pdf", "chunk_index": i}))
    )

conn.commit() # Save all inserted rows

# ---  Retrieval Function ---
def retrieve_context(query, k=5):
    embedding = get_embedding(query)
    vector_str = f"[{','.join(map(str, embedding))}]"
    cursor.execute(
        "SELECT content FROM table ORDER BY embedding <=> %s LIMIT %s",
        (vector_str, k)
    )
    return [row[0] for row in cursor.fetchall()]

# --- Generation Function ---

# For Gemini integration:
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# model = genai.GenerativeModel("gemini-2.5-pro-preview-06-05")

def generate_answer(query):
    context = retrieve_context(query)
    context_text = "\n\n".join(context)

    prompt = f"""Answer the two following questions based on the retrieved passages provided from each query.

1. Relevance Evaluation  
Does the retrieved context directly pertain to the topic and scope of the query?

Provide a relevance score between 0 and 100, where:

100 = The context is entirely focused on the query's subject  
50 = The context is partially related (e.g. correct company but wrong financial quarter)  
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
{context_text}

Query: {query}
Answer:
"""
    # --- For Gemini integration ---
    # response = model.generate_content(prompt)
    # return response.text, context

    # --- For GPT generation ---
    response = openai_client.chat.completions.create(
        model="gpt-4o",  # You can change to gpt-3.5-turbo
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content, context

# --- Read Queries and Generate Results ---
queries_df = pd.read_csv(query_csv_path)  # Read queries from the specified CSV file
os.makedirs("results", exist_ok=True)

with open(output_csv_path, "w", newline="") as f: # Open a new CSV to store the generated answers
    writer = csv.DictWriter(f, fieldnames=["query", "generated_answer", "passage"])
    writer.writeheader()

    for q in queries_df["query"]:
        try:
            print(f"Processing: {q}")
            answer, context = generate_answer(q) # Generate a score for the query
            writer.writerow({
                "query": q,
                "generated_answer": answer,
                "passage": "\n\n====================\n\n".join(context) # Join passages with separator
            })
        except Exception as e:
            print(f"Error for query '{q}':", e)
