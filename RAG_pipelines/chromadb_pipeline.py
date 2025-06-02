# Build your own powerful RAG pipeline!
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system with ChromaDB.

#--- Installation ---
# First, make sure you have these essential libraries installed.
# You can run this command in your terminal:
# pip install openai chromadb langchain-community pypdf pandas sentence-transformers

# --- Import Libraries ---
import os
import csv
import pandas as pd
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- Configuration ---
OPENAI_API_KEY = "your-api-key"  # Replace this with your OpenAI API key
client = OpenAI(api_key=OPENAI_API_KEY) # Initialize the OpenAI client

pdf_path = "your-pdf-path"  # Path to your PDF document
query_csv_path = "your-query-file" # Path to your CSV file containing queries
output_csv_path = "results/results.csv" # Path to save the output CSV with results

# --- Load and Chunk PDF ---
loader = PyPDFLoader(pdf_path) # Load the PDF document
pages = loader.load() # Load all pages from the PDF
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # Initialize text splitter
chunks = splitter.split_documents(pages) # Split the PDF pages into smaller chunks

# --- Initialize Embedder ---
embedder = SentenceTransformer("all-MiniLM-L6-v2") # Initialize a sentence transformer model for creating embeddings

# --- Initialize ChromaDB Client and Collection ---
client_chroma = chromadb.PersistentClient(path="chromadb_data") # Initialize a persistent ChromaDB client
if "rag_docs" in [c.name for c in client_chroma.list_collections()]: # Check if the collection already exists
    client_chroma.delete_collection("rag_docs") # If it exists, delete it to start fresh
collection = client_chroma.create_collection(name="your-name") # Create a new collection 

# --- Add Chunks to ChromaDB ---
for i, chunk in enumerate(chunks): # Iterate through each text chunk
    text = chunk.page_content.strip() # Extract the text content from the chunk
    emb = embedder.encode(text).tolist() # Generate embeddings for the text chunk
    collection.add(
        ids=[str(i)], # Assign a unique ID for each chunk
        documents=[text], # Store the text content
        embeddings=[emb] # Store the generated embeddings
    )

# --- Retrieval Function ---
def retrieve_context(query, k=5):
    query_emb = embedder.encode(query).tolist() # Generate an embedding for the input query
    results = collection.query(query_embeddings=[query_emb], n_results=k) # Query ChromaDB for similar documents
    return results["documents"][0] if results["documents"] else ["[No context retrieved]"] # Return the retrieved documents

# --- Generation Function ---
def generate_answer(query):
    context = retrieve_context(query) # Retrieve relevant context for the query
    prompt = f"""Answer the following question using the provided context.

Context:
{chr(10).join(context)}

Question: {query}
Answer:""" # Construct the prompt for the language model
    response = client.chat.completions.create(
        model="gpt-4o", # Specify the OpenAI model to use
        messages=[{"role": "user", "content": prompt}], # Pass the prompt as a user message
        temperature=0.2 # Set the creativity level of the model (lower for more factual)
    )
    return response.choices[0].message.content, context # Return the generated answer and the context used

# --- Read Queries and Generate Answers ---
queries_df = pd.read_csv(query_csv_path) # Read queries from the specified CSV file
os.makedirs("results", exist_ok=True) # Create a directory for results 

with open(output_csv_path, "w", newline="") as f: # Open the output CSV file for writing
    writer = csv.DictWriter(f, fieldnames=[ "query", "generated_answer", "passage"]) # Initialize CSV writer with specified fieldnames
    writer.writeheader() # Write the header row to the CSV

    for q in queries_df["query"]: # Iterate through each query in the DataFrame
        print(f"Processing: {q}") # Print the current query being processed
        try:
            answer, context = generate_answer(q) # Generate an answer and retrieve context for the query
            writer.writerow({
                "query": q,
                "generated_answer": answer,
                "passage": " ".join(context) # Write the query, generated answer, and contexts to the CSV
            })
        except Exception as e: # Catch any exceptions during processing
            print(f"Error for query '{q}':", e) # Print an error message if an exception occurs
