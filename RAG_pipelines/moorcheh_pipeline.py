# Build your own powerful RAG pipeline! 
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system with Moorcheh.

#--- Installation ---
# First, make sure you have these essential libraries installed.
# You can run this command in your terminal:
# pip install openai moorcheh-sdk langchain_community pypdf pandas

#--- Import Libraries ---
import os
import csv
import pandas as pd
from openai import OpenAI
from moorcheh_sdk import MoorchehClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
OPENAI_API_KEY = "your-openai-apikey"  # Input your OpenAI API key here
MOORCHEH_API_KEY = "your-moorcheh-apikey"  # Input your Moorcheh API key here
os.environ["MOORCHEH_API_KEY"] = MOORCHEH_API_KEY # Set Moorcheh key in environment

pdf_path = "your-pdf-path"  # Path to your PDF document
query_csv_path = "your-query-path" # Path to your CSV file with queries
output_csv_path = "results/results.csv" # Where to save the results

# --- Initialize Clients ---
openai_client = OpenAI(api_key=OPENAI_API_KEY) # Connect to OpenAI
moorcheh_client = MoorchehClient() # Connect to Moorcheh vector database

# --- Load and Chunk PDF ---
loader = PyPDFLoader(pdf_path) # Load your PDF file
pages = loader.load() # Get all pages from the PDF

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) # Split text into smaller pieces
chunks = splitter.split_documents(pages) # Divide PDF pages into chunks

# --- Create Moorcheh Namespace ---
namespace_name = "your-namespace-name" # Set the name for your new namespace
moorcheh_client.create_namespace(namespace_name= namespace_name, type="text") # Create a text-based namespace for your data in Moorcheh

# --- Upload Chunks to Moorcheh ---
for i, chunk in enumerate(chunks): # Go through each text chunk
    text = chunk.page_content.strip() # Get the text from the chunk
    document = { # Prepare chunk for Moorcheh
        "id": f"chunk_{i}", # Unique ID for each chunk
        "text": text, # The actual text content
        "metadata": {"source": "pdf", "chunk_index": i} # Extra info about the chunk
    }
    moorcheh_client.upload_documents(namespace_name=namespace_name, documents=[document]) # Upload chunk to Moorcheh

# --- Retrieval Function ---
def retrieve_context(query, k=5): # Function to find relevant text for a query
    results = moorcheh_client.search( # Search Moorcheh for similar text
        namespaces=[namespace_name], # Search in our document collection
        query=query, # The question we want to answer
        top_k=k # How many top results to get
    )
    matches = results.get("matches", []) or results.get("results", []) # Get the search results
    return [hit["text"] for hit in matches] if matches else ["[No context retrieved]"] # Return the relevant text

# --- Generation Function ---
def generate_passage(query): # Function to create an answer from context
    context = retrieve_context(query) # Get relevant info for the question
    prompt = f"""Answer the following question using the provided context.

Context:
{chr(10).join(context)}

Question: {query}
Answer:""" # Build the prompt for OpenAI
    response = openai_client.chat.completions.create( # Ask OpenAI to generate an answer
        model="gpt-4o", # Using the GPT-4o model
        messages=[{"role": "user", "content": prompt}], # Your question for the AI
        temperature=0.2 # How creative the AI should be (lower is less creative)
    )
    return response.choices[0].message.content, context # Return the AI's answer and the context used

# --- Read Queries and Generate Results ---
queries_df = pd.read_csv(query_csv_path) # Load your questions from a CSV file
os.makedirs("results", exist_ok=True) # Create a folder for results if it doesn't exist

with open(output_csv_path, "w", newline="") as f: # Open the results CSV file
    writer = csv.DictWriter(f, fieldnames=["passage_id", "query", "generated_answers", "passage"]) # Set up CSV columns
    writer.writeheader() # Write the column headers

    for idx, q in enumerate(queries_df["query"]): # Go through each question
        print(f"Processing: {q}") # Show which question is being processed
        try:
            passage, context = generate_passage(q) # Get AI's answer and context
            writer.writerow({ # Write the results to the CSV
                "passage_id": idx, # Unique ID for this answer
                "query": q, # The original question
                "generated_answers": passage, # The AI-generated answer
                "passage": " ".join(context) # All contexts used, joined into one string
            })
        except Exception as e: # If something goes wrong
            print(f"Error for query '{q}':", e) # Print the error
