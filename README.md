# Moorche Benchmarks

**Evaluate and compare semantic search accuracy of Moorche and other AI models.**

It is often necessary to test and evaluate AI models to ensure that they are providing the correct and meaningful answers. This repo provides RAG pipelines to test various AI models, including Moorche for their response accuracy

## Key Features
Ease of use, all pipelines share a similar design
Metrics, generated output files contain the generated answers as well as the relevant chunks used
## Getting Started Guide
This guide walks you through how to use the RAG pipelines. We will be using Moorche as an example
### Prerequisites
Python: Version 3.9 or higher
OpenAI API Key, set this as an environment variable on your device
Optional: Vectara Account, If you would like to copy the output file into a generated_answers file for evaluation with Open RAG Eval
### Installation
Install the required python packages listed in the comments at the top file containing the pipeline. For Moorche it is:
'''pip install openai moorcheh-sdk langchain_community pypdf pandas'''
