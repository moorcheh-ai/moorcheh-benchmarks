# Build your own powerful RAG pipeline!
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system using pgvector.

# --- Installation ---
# Before you begin, install these libraries using pip:
# pip install openai langchain_community pypdf pandas psycopg2-binary python-dotenv

# --- Import Libraries ---
import os
import csv
import json
import psycopg2
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Load API keys and environment variables from a .env file ---
load_dotenv()

# --- Configuration ---
# Set your OpenAI API key
OPENAI_API_KEY = "your-openai-apikey"  # Replace with your OpenAI API key
openai_client = OpenAI(api_key=OPENAI_API_KEY)  # Initialize OpenAI client

# Connect to your local PostgreSQL database with pgvector installed
conn = psycopg2.connect(
    dbname="ragdb",           # Your Postgres database name
    user="your-username",     # Your database username
    password="your-password", # Your password (leave blank if not needed)
    host="localhost",         # Host for local DB
    port="5432"               # Default Postgres port
)
cursor = conn.cursor()

# --- File Paths ---
pdf_path = "path-to-your-pdf.pdf"            # Input PDF file with source knowledge
query_csv_path = "your-query-file.csv"       # CSV containing user queries
output_csv_path = "results/results.csv"      # Where to store final results

# --- Load and Chunk PDF ---
loader = PyPDFLoader(pdf_path)               # Load PDF using LangChain's PyPDFLoader
pages = loader.load()                        # Get all the pages as documents

# Split the documents into manageable chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Max characters per chunk
    chunk_overlap=50       # Overlap to preserve context between chunks
)
chunks = splitter.split_documents(pages)     # Apply chunking

# --- Generate Embeddings and Upload Chunks to pgvector ---
def get_embedding(text):
    # Use OpenAI to convert text into dense vector embedding
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",  
        input=text
    )
    return response.data[0].embedding

# Loop through each chunk and insert it into the database
for i, chunk in enumerate(chunks):
    text = chunk.page_content.strip()  # Clean the chunk text
    embedding = get_embedding(text)    # Convert chunk into embedding

    # Insert into your rag_chunks table (must have columns: chunk_index, content, embedding, metadata)
    cursor.execute(
        "INSERT INTO rag_chunks (chunk_index, content, embedding, metadata) VALUES (%s, %s, %s, %s)",
        (i, text, embedding, json.dumps({"source": "pdf", "chunk_index": i}))
    )

conn.commit()  # Save all inserted rows

# ---  Retrieval Function ---
def retrieve_context(query, k=5):
    embedding = get_embedding(query)

    vector_str = f"[{','.join(map(str, embedding))}]"

    # Use pgvector's operator to find the top-k most similar chunks
    cursor.execute(
        f"SELECT content FROM rag_chunks ORDER BY embedding <=> %s LIMIT %s",
        (vector_str, k)
    )

    # Return just the text content of the top-k chunks
    return [row[0] for row in cursor.fetchall()]

# --- Answer Generation Function ---
def generate_passage(query):
    # Retrieve relevant chunks from the vector database
    context = retrieve_context(query)

    # Construct a prompt that feeds the context + query to the LLM
    prompt = f"""Answer the question using the provided context.

Context:
{chr(10).join(context)}

Question: {query}
Answer:"""

    # Ask OpenAI GPT-4o to generate a response based on the context
    response = openai_client.chat.completions.create(
        model="gpt-4o",  # You can use gpt-4, gpt-3.5-turbo, etc.
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # Lower temperature = more factual, less creative
    )

    # Return the model's response + context used
    return response.choices[0].message.content, context

# --- Read Queries and Generate Results ---
# Read the list of questions from your CSV
queries_df = pd.read_csv(query_csv_path)
os.makedirs("results", exist_ok=True)  # Create output folder if it doesn't exist

# Open a new CSV to store the generated answers
with open(output_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["passage_id", "query", "generated_answer", "passage"])
    writer.writeheader()  # Write column headers

    # Loop through each query and generate a response
    for idx, q in enumerate(queries_df["query"]):
        print(f"Processing: {q}")
        try:
            answer, context = generate_passage(q)  # Run the pipeline
            writer.writerow({
                "passage_id": idx,
                "query": q,
                "generated_answer": answer,
                "passage": "".join(context)  # Join context into one string
            })
        except Exception as e:
            print(f"Error for query '{q}':", e)  # Show any runtime errors
