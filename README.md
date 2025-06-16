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
Upload a document to the same folder that the pipeline is in. Set the value of the `pdf_path` variable to be the name of the document. \n
Change the chunking method in the pipeline to your desired chunking method.
### Running and Output
Run the program. It should produce an output file with the `passage_id`, `query`, `generated_answer`, and `passage`. By default, the file is called `results.csv` and is found in a created `results` folder. A sample snippet of an output file is shown below:
```
passage_id,query,generated_answers,passage
0,Tesla market penetration opportunities existing markets.,"1. Relevance Evaluation

Relevance Score: 70

Rationale: The context is partially related to the query about Tesla's market penetration opportunities in existing markets. It discusses Tesla's strategic considerations and market positioning, such as its focus on premium and luxury markets, and its potential growth in developing economies like India and Southeast Asia. However, it does not specifically address market penetration opportunities in existing markets, such as the U.S. or Europe, where Tesla is already established. The context also includes information about Tesla's energy storage business, which is not directly relevant to the query.

2. Completeness Evaluation

Completeness Score: 50

Rationale: The context provides some information relevant to Tesla's market penetration opportunities, particularly in emerging markets. It highlights Tesla's strategic dilemma regarding its market focus and the potential for growth in developing economies. However, it lacks specific details about market penetration strategies or opportunities in existing markets where Tesla is already present. Key information, such as specific strategies for increasing market share or overcoming challenges in established markets, is missing.","Labor Costs and Supply Chain Economics
Tesla’s vertically integrated model helps control some costs,
but global labor markets still influence operations. For
example, manufacturing in Germany involves higher labor
costs compared to China. Yet, the proximity to European
markets and skilled labor justifies the investment. Tesla also
benefits from economies of scale, reducing unit costs as
production ramps up.
```
## Evaluation
To evaluate the relevance and completeness scores of the AI models, it is recommended that you paste the scores into a spreadsheet
