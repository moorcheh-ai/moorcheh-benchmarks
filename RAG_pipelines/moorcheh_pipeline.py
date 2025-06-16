# Build your own powerful RAG pipeline! 
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system with Moorcheh.

#--- Installation ---
# First, make sure you have these essential libraries installed.
# You can run this command in your terminal:
# pip install openai moorcheh-sdk langchain_community pypdf pandas

#--- Import Required Libraries ---
import os
import csv
import pandas as pd
from openai import OpenAI
from moorcheh_sdk import MoorchehClient
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
<<<<<<< HEAD
from google import genai
import time

genai_client = genai.Client(api_key="AIzaSyC87lSZT2G-66dxnHz9xljWwIcprf79iHU")
# 
# --- Configuration ---
OPENAI_API_KEY = "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"  # Input your OpenAI API key here
MOORCHEH_API_KEY = "spEC1UErKk7bc8kPUtLkoYX00GNSe3J9ofaM3p6a"  # Input your Moorcheh API key here
=======
from sentence_transformers import SentenceTransformer  # Optional: Can be used for embedding

#--- Configuration ---
OPENAI_API_KEY = "your-openai-apikey"  # Input your OpenAI API key here
MOORCHEH_API_KEY = "your-moorcheh-apikey"  # Input your Moorcheh API key here
>>>>>>> 2919ac9171e7d8ce400afb8253a6704baafe71d7
os.environ["MOORCHEH_API_KEY"] = MOORCHEH_API_KEY # Set Moorcheh key in environment

pdf_path = "merged-pdf.pdf"  # Path to your PDF document
query_csv_path = "queries.csv" # Path to your CSV file with queries
output_csv_path = "results/results.csv" # Where to save the results
top_k = 5  # Number of documents to retrieve

# --- Initialize Clients ---
client = OpenAI(api_key=OPENAI_API_KEY) # Connect to OpenAI
moorcheh_client = MoorchehClient() # Connect to Moorcheh vector database

# --- Load and Chunk PDF ---
loader = PyPDFLoader(pdf_path) # Load your PDF file
pages = loader.load() # Get all pages from the PDF

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120) # Split text into smaller pieces
chunks = splitter.split_documents(pages) # Divide PDF pages into chunks

# --- Create Moorcheh Namespace ---
namespace_name = "rag_docs" # Set the name for your new namespace
# moorcheh_client.create_namespace(namespace_name= namespace_name, type="text") # Create a text-based namespace for your data in Moorcheh

#--- Upload Chunks with Metadata to Moorcheh ---
documents = []
for i, chunk in enumerate(chunks): # Go through each text chunk
    text = chunk.page_content.strip() # Get the text from the chunk
    doc = { # Prepare chunk for Moorcheh
        "id": f"chunk_{i}", # Unique ID for each chunk
        "text": text, # The actual text content
        "metadata": {"source": "pdf", "chunk_index": i} # metadata
    }
<<<<<<< HEAD
    time.sleep(0.2)
    moorcheh_client.upload_documents(namespace_name=namespace_name, documents=[document]) # Upload chunk to Moorcheh
=======
    documents.append(doc)
>>>>>>> 2919ac9171e7d8ce400afb8253a6704baafe71d7

# Batch Upload the pdf
batch_size = 15
for start in range(0, len(documents), batch_size):
    moorcheh_client.upload_documents(
        namespace_name=namespace_name,
        documents=documents[start:start + batch_size]
    )

#--- Retrieve Top-k Contexts with Scores ---
def retrieve_context(query, k=top_k):
    results = moorcheh_client.search(
        namespaces=[namespace_name],
        query=query,
        top_k=k
    )
    matches = results.get("matches", []) or results.get("results", [])
    
    # Return text, score, and metadata for each match
    return [
        {
            "text": m["text"],
            "score": m.get("score", 0),
            "metadata": m.get("metadata", {})
        }
        for m in matches
    ]

#--- Generation Function ---
def generate_answer(query): # Function to create an answer from context
    context_with_scores = retrieve_context(query) # Get relevant info for the question
    context_text = "\n\n".join(
    f"[Score: {round(ctx['score'], 3)}]\n{ctx['text']}" for ctx in context_with_scores
    )

<<<<<<< HEAD
# --- Generation Function ---
def generate_passage(query): # Function to create an answer from context
    context = retrieve_context(query) # Get relevant info for the question
    prompt = f"""Answer the two following questions based on the retrived passages provided from each query.      1. Relevance Evaluation
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
=======
    prompt = f"""Answer the following question using the provided context.
Each passage is prefixed with its relevance score.
>>>>>>> 2919ac9171e7d8ce400afb8253a6704baafe71d7

Context:
{context_text}

Question: {query}
Answer:"""
<<<<<<< HEAD
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content, context
=======

    response = openai_client.chat.completions.create( # Ask OpenAI to generate an answe
        model="gpt-4o", # Using the GPT-4o model
        messages=[{"role": "user", "content": prompt}], # Your question for the AI
        temperature=0.2 # How creative the AI should be (lower is less creative)
    )
    return response.choices[0].message.content, [ctx['text'] for ctx in context_with_scores]
>>>>>>> 2919ac9171e7d8ce400afb8253a6704baafe71d7

#--- Run All Queries and Save Results ---
queries_df = pd.read_csv(query_csv_path) # Load your questions from a CSV file
os.makedirs("results", exist_ok=True) # Create a folder for results if it doesn't exist

    
with open(output_csv_path, "w", newline="") as f: # Open the results CSV file
    writer = csv.DictWriter(f, fieldnames=["passage_id", "query", "generated_answers", "passage"]) # Set up CSV columns
    writer.writeheader() # Write the column headers

    for idx, q in enumerate(queries_df["query"]): # Go through each question
        print(f"Processing: {q}") # Show which question is being processed
        try:
            passage, context = generate_answer(q) # Get AI's answer and context
            writer.writerow({ # Write the results to the CSV
                "passage_id": idx, # Unique ID for this answer
                "query": q, # The original question
                "generated_answers": passage, # The AI-generated answer
                "passage": " ".join(context) # All contexts used, joined into one string
            })
        except Exception as e: # If something goes wrong
            print(f"Error for query '{q}':", e) # Print the error
