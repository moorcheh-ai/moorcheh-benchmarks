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
What is Meta’s Net Income in 2025?
What is Netflix’s Profit in 2025?
What is Apple’s Expenses in 2025?
What is Alphabet’s Revenue in 2025?
What is Amazon’s operating expenses in 2025?
What is Alphabet’s Revenue in 2025?
Which company had the highest revenue in the first quarter of 2025?
Which company has seen the least growth?
Which of the five companies showed the strongest overall financial performance in early 2025 and why?
Compare the quarter-on-quarter revenue growth between Meta and Alphabet.
```
### Documents and Chunking
Upload a document to the same folder that the pipeline is in. Set the value of the `pdf_path` variable to be the name of the document. \n
Change the chunking method in the pipeline to your desired chunking method.
### Running and Output
Run the program. It should produce an output file with the `passage_id`, `query`, `generated_answer`, and `passage`. By default, the file is called `results.csv` and is found in a created `results` folder. A sample snippet of an output file is shown below:
```
passage_id,query,generated_answer,passage
0,What is Meta’s Net Income in 2025?,"Meta’s Net Income in the first quarter of 2025 is $16,644 million.","META PLATFORMS, INC.
CONDENSED CONSOLIDATED STATEMENTS OF INCOME
(In millions, except per share amounts)
(Unaudited)
Three Months Ended March 31,
2025 2024
Revenue $ 42,314 $ 36,455 
Costs and expenses:
Cost of revenue  7,572  6,640 
Research and development  12,150  9,978 
Marketing and sales  2,757  2,564 
General and administrative  2,280  3,455 
Total costs and expenses  24,759  22,637 
Income from operations  17,555  13,818 
Interest and other income, net  827  365 Meta Reports First Quarter 2025 Results
MENLO PARK, Calif. – April 30, 2025  – Meta Platforms, Inc. (Nasdaq: META) today reported financial results for the 
quarter ended March 31, 2025.
""We've had a strong start to an important year, our community continues to grow and our business is performing very well,"" 
said Mark Zuckerberg, Meta founder and CEO. ""We're making good progress on AI glasses and Meta AI, which now has 
almost 1 billion monthly actives.""
First Quarter 2025 Financial Highlights META PLATFORMS, INC.
CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS
```
## Optional: Integration with Open RAG Eval
You can use the output file directly with [Open RAG Eval](https://github.com/vectara/open-rag-eval?tab=readme-ov-file). Simply copy the contents of the file and paste it into the answers csv file and run Open RAG eval to get an output file with all the scores.
Open RAG Eval uses the [UMBRELA](https://arxiv.org/pdf/2406.06519) and [AutoNuggetizer](https://arxiv.org/pdf/2411.09607) methods to score the accuarcy of the responses.
In brief, UMBRELA uses an LLM to understand the intent behind the queries and passages and label said passages with different relevance scores [^1]. On the other hand, AutoNuggetizer is a framework that uses LLMs to analyze "nuggets", which are discrete pieces of facts drawn from either the documents or the generated answers. It compares the nuggets from documents and the nuggets from generated answers and finds matches or mismatches, which contributes to a score. Using the the output file from Open RAG Eval, you can then measure the accuracy of the AI models using their testing methodology.

[^1]:Lin, J., Upadhyay, S., Pradeep, R., Thakur, N., & Craswell, N. (2024, June 10). UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor. arxiv.org. https://arxiv.org/pdf/2406.06519 
[^2]: Lin, J., Upadhyay, S., Pradeep, R., Thakur, N., Craswell, N., & Campos, D. (2024, November 14). Initial Nugget Evaluation Results for the TREC 2024 RAG Track with the AutoNuggetizer Framework. arxiv.org. https://arxiv.org/pdf/2411.09607 
