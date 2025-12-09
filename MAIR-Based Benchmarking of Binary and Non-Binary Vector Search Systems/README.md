# **MAIR-Based Benchmarking of Binary and Non-Binary Vector Search Systems**

This repository contains a comprehensive Google Colab workflow for benchmarking multiple vector search systems using the **MAIR (Massive AI Retrieval)** and **BEIR** datasets. It evaluates both **binary (1-bit)** and **non-binary (dense float)** vector search approaches across leading vector databases and search engines.

The notebook provides an end-to-end pipeline for:
* Downloading, organizing, and unifying MAIR/BEIR datasets.
* Generating embeddings using **Cohere (v4.0)** and converting them for binary retrieval.
* Running systematic benchmarks across **Moorcheh**, **Pinecone**, **Elasticsearch**, and **PostgreSQL (PGVector)**.
* Comparing latencies, retrieval quality (NDCG, MAP, Recall), storage efficiency, and indexing speed.

---

## **Contents**

### **1. Dataset Download and Organization**
* **Sources:** Supports downloading datasets from **MAIR-Bench** (Hugging Face) and standard **BEIR** benchmarks.
* **Processing:** Automatically organizes corpus documents, queries, and relevance judgments (qrels) into a unified directory structure within Google Drive (`docs` and `queries` folders).
* **Category Detection:** Automatically categorizes MAIR datasets into domains (Legal, Medical, Financial, Code, etc.) for easier selection.

### **2. Binary Vector Search Benchmarking**
Evaluates **1-bit quantified vectors** (Sign-based Binarization: values $\ge$ 0 $\to$ 1, < 0 $\to$ 0).

* **Moorcheh (Binary Mode):** Native binary vector search.
* **Pinecone (Binary Index):** Cosine similarity on binary vectors.
* **Elasticsearch:** Dense vector field optimization for binary data.
* **PGVector / PostgreSQL:**
    * Compiles `pgvector` from source within Colab to enable advanced features.
    * Uses the `BIT` data type for 32x storage compression.
    * Implements **HNSW index with Hamming distance** (`bit_hamming_ops`).
* **Features:** Calculates storage savings (compression ratio) and exports binarized embeddings in multiple formats (`.npy`, `.csv`, `.json`, `.h5`, `.parquet`).

### **3. Dense (Non-Binary) Vector Search Benchmarking**
Evaluates standard **float32** vector search performance.

* **Moorcheh (Dense Mode):** Standard dense vector search.
* **Pinecone:** Dense index with optional **Cohere Reranking (v3.5)** step.
* **Elasticsearch:** Uses `dense_vector` fields with Cosine kNN search.
* **PGVector / PostgreSQL:** Uses `vector` data type with HNSW indexing (`vector_cosine_ops`).

### **4. Evaluation Metrics**
The notebook computes a detailed performance report including:
* **Retrieval Quality:** NDCG@k, MAP@k, Recall@k, Precision@k (k = 1, 3, 5, 10, 100).
* **Latency:** Mean, Median, Min, Max, and P99 timings for Client-side vs. Server-side search.
* **Indexing Performance:** Total ingestion time and indexing overhead.
* **Storage Efficiency:** Comparison of float vs. binary storage footprints.

---

## **Requirements**

The notebook is optimized for **Google Colab** and automatically installs the specific dependencies required for each provider, including:

* **Core:** `transformers`, `datasets`, `pandas`, `numpy`, `tqdm`, `beir`
* **Vector DB Clients:** `moorcheh-sdk`, `pinecone-client`, `elasticsearch`
* **PostgreSQL:** `psycopg2-binary`, `pgvector` (Python client), and system-level PostgreSQL installation commands.
* **Embeddings:** `cohere` (SDK)

### **API Keys Required**
You must provide API keys via Google Colab Secrets or Environment Variables:
* `COHERE_API_KEY` (Required for embedding generation)
* `MOORCHEH_API_KEY`
* `PINECONE_API_KEY`
* `ELASTIC_URL` / `ELASTIC_API_KEY` (or username/password)

---

## **How to Use**

1.  **Setup:** Open the Colab notebook and mount Google Drive (handled in the first cell).
2.  **Data Prep:** Run the **Dataset Download** section to fetch and combine MAIR/BEIR data.
3.  **Select Benchmark:**
    * **Binary Benchmarking:** Choose datasets and providers. The script will generate embeddings, binarize them, upload them to the selected DBs, and run search tests.
    * **Non-Binary Benchmarking:** Similar workflow using standard float embeddings, with optional reranking steps.
4.  **Analyze Results:**
    * Real-time logs show upload/search speeds.
    * Final results are summarized in ASCII tables and saved to CSV files in your Google Drive (e.g., `MAIR.PGVector.Binary.Cohere.V4.csv`).

---

## **Key Features**

* **Apple-to-Apple Comparison:** Generates embeddings **once** per dataset and reuses them across all providers to ensure fair comparison.
* **PGVector Compilation:** The notebook includes a script to compile `pgvector` from source, ensuring access to the latest `BIT` type optimizations not always available in standard `apt` packages.
* **Interactive Selection:** Users can interactively select specific datasets (by ID or category) and specific providers to test during runtime.
* **Automated Cleanup:** Options to automatically delete namespaces/indices after testing to prevent accruing cloud costs.

---

## **License**

This benchmarking suite is provided for research and experimentation purposes. Please adhere to the usage policies of the individual service providers (Cohere, Pinecone, Moorcheh) and the licenses of the datasets used (MAIR/BEIR).