# Moorche Benchmarks

**Evaluate and compare semantic search accuracy of Moorche and other AI models.**

It is often necessary to test and evaluate AI models to ensure that they are providing the correct and meaningful answers. This repo provides RAG pipelines to test various AI models, including Moorche for their response accuracy

## Key Features
- Ease of use, all pipelines share a similar design
- Metrics, generated output files contain the generated answers as well as the relevant chunks used
## Getting Started Guide
This guide walks you through how to use the RAG pipelines. We will be using Moorche as an example and the experiment will involve asking queries involving FAANG's earnings releases.
### Prerequisites
Python: Version 3.9 or higher
OpenAI API Key, set this as an environment variable on your device
Optional: Vectara Account, If you would like to copy the output file into a generated_answers file for evaluation with Open RAG Eval
### Installation
Install the required python packages listed in the comments at the top file containing the pipeline. For Moorche it is:
```
pip install openai moorcheh-sdk langchain_community pypdf pandas
```
### Queries
Create a csv file in the same folder that the pipeline is in to hold your queries. By default, the pipeline will look for a file named queries.csv, but you can change this in the code if your file uses a different name. A sample queries.csv file is shown below:
```
query
Tesla market penetration opportunities existing markets.
Current market share of Tesla in existing markets.
Customer segments in existing markets for Tesla services.
Competitive analysis of Tesla's position in existing markets. 
Tactical recommendations for increasing Tesla's market share. 
Indicators for Tesla's expansion into new geographic regions.
Market readiness indicators for Tesla's existing offerings.
Industry trends signaling opportunities for Tesla expansion.
Competitive landscape analysis for Tesla in new geographic regions.
```
### Documents and Chunking
Upload a document to the same folder that the pipeline is in. Set the value of the `pdf_path` variable to be the name of the document. 
Change the chunking method in the pipeline to your desired chunking method.

### Prompt
The current prompt,
```
Answer the two following questions based on the retrived passages provided from each query.
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

    50 = The context includes some, but not all, key information (e.g., only Q1 revenue when the query asks for full-year revenue)

    0 = The context includes none of the necessary information to answer the query

    Rationale:
    Clearly state whether the context contains the required facts, figures, or explanations needed to construct a complete answer. If any crucial components are missing, specify what they are.
```
is used to benchmark each of the AI models. For the rest of this document, we will be using this prompt to explain our evaluation methodology. Feel free to change the prompt to your own for your experiments.

### Running and Output
Run the program. It should produce an output file with the `passage_id`, `query`, `generated_answer`, and `passage`. By default, the file is called `results.csv` and is found in a created `results` folder. A sample snippet of an output file is shown below:
```
 ### 1. Relevance Evaluation
 
 **Relevance Score: 100**
 
 **Rationale:**
 The context is entirely focused on the query's subject. It directly discusses various factors that serve as "market readiness indicators" for Tesla. These include:
 *  **Competitive Landscape:** Mentions "competitive responses from established automakers" and tax credits for competitors.
 *  **Public Perception:** Introduces the "Tesla–Musk Image Score" to quantify public perception and consumer sentiment.
 *  **Infrastructure:** Highlights the "widespread availability of Tesla Supercharger stations" and a metric called the "Supercharge Station Ratio."
 *  **Market Performance:** Provides specific growth figures for Tesla's energy storage offerings (Megapack).
 *  **External Factors:** Lists social, technological, and legal factors such as sustainability values, tech-savvy demographics, and regulatory compliance.
 
 Every passage contributes directly to understanding the market conditions and readiness for Tesla's products.

 ### 2. Completeness Evaluation
 
 **Completeness Score: 85**
 
 **Rationale:**
 The context provides a very strong and multi-faceted answer. It successfully identifies a wide range of market readiness indicators, from infrastructure and competition to public sentiment and actual product growth. Crucially, it provides a specific, quantitative example of market uptake with the energy storage growth figures ("up 114% year over year").

**Retrieved Passages:**
[Score: 0.087]
 minants of Tesla’s market share, focusing on factors such as competitive responses from
 established automakers, product differentiation strategies, and the impact of public policies.
 This model allows us to quantitatively assess the strength of Tesla’s competitive advantage
 over time and how second movers have systematically narrowed the gap. Second, we
 introduce a heterogeneous effects framework with an interaction model, exploring how
 key external factors—such as tax credits and consumer sentiment—unequally influence
 Tesla’s market performance. A novel contribution is the development of the Tesla–Musk
 Image Score, an innovative index that quantifies the evolving public perception of Tesla
 ====================
...
```


## Evaluation Methodology
We evaluated the vector databases using a custom methodology to assess **relevance** and **completeness** of the retrieved document chunks. The evaluation process controlled for multiple factors to ensure fair and consistent results.

### Key Controls in the Evaluation:
Key Controls in the Evaluation:
1. **Standardized Queries**: All vector databases were tested using the same set of **queries**. This eliminated any bias that could be introduced by varying query types.

2. **Document Chunking**: Documents were consistently chunked using a **700-character chunk size** with **120-character overlap**, ensuring uniform chunking across all systems.

3. **LLM-based Scoring**: We used **ChatGPT-mini-4o** and **Gemini-2.5-pro** as the judges for evaluating the relevance and completeness of each retrieved chunk. Both models were evaluated with the same scoring prompt, ensuring consistency across all evaluations.

4. **Scoring Criteria**:

  - **Relevance**: How closely the retrieved chunk matches the query's topic.

    - 100 = Fully relevant

    - 50 = Partially relevant

    - 0 = Not relevant

  - **Completeness**: How well the retrieved chunk answers the query.

    - 100 = Fully complete

    - 50 = Partially complete

    - 0 = Incomplete

5. Prompt: The same prompt was used across all models for relevance and completeness evaluation:

  - **Relevance**: "Does the retrieved context directly pertain to the topic and scope of the query?"

  - **Completeness**: "If someone were to answer the query using only this context, how complete and sufficient would their answer be?"

6. **Result Aggregation**: After assigning relevance and completeness scores to each retrieved chunk, the scores were aggregated by **averaging the scores** across all queries for each vector database in a spreadsheet. This ensures that the final comparison was based on an overall performance across all queries.

## Results
The results of our experiment show that Moorcheh scores the highest in relevance and completeness overall. It especially excels in more difficult, comprehensive questions like the query "Indicators for Tesla's expansion into new geographic regions". We recommend that you paste all your results from the results.csv file into a spreadsheet like Excel or Google Sheets for ease of comparison and creating graphs to display the data. We have a report based on our experiment in a [discussion post](https://github.com/mjfekri/moorche-benchmarks/discussions/3). Feel free to change the queries, documents, chunking methods, and prompts to fit your own experiments.
