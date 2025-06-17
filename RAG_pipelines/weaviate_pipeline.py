# Build your own RAG pipeline! 
# This guide walks you through creating a Retrieval Augmented Generation (RAG) system with Weviate.

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
import weaviate
from weaviate.auth import AuthApiKey
import uuid
import google.generativeai as genai   # Optional, for future integration with Gemini

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
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
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
                "passage": "\n\n====================\n\n".join(context)
            })
        except Exception as e:
            print(f"Error for query '{q}':", e)

client_wv.close()