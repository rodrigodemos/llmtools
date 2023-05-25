# LLMTools

This repo is a collection of Azure examples on LLM


## Prerequisites

* Python 3.10
* Visual Studio Code
* Azure Subscription
* Azure Open AI
* Azure Form Recognizer
* Azure Cache for Redis (Enteprise with RediSearch module)

## How to use

1. Clone this repo (into your local)
1. Open project with vscode
1. Open a terminal in vscode and create a python virtual environment
1. Activate virtual environment
1. Install required libraries
1. Rename sample.env to .env
1. Update .env with your environment info

```
git clone https://github.com/rodrigodemos/llmtools c:\repos\llmtools

cd c:\repos\llmtools

code .

python -m venv .venv

.venv\scripts\activate

pip install -r requirements.txt

ren sample.env .env
```

## extract_and_upload.py

This script is doing the following:
* Loads a local PDF file into Azure Form Recognizer
* Builds a list of paragraphs from the results and creates additional attributes based on document headings and sentence format (i.e TOPIC: This sentence is related to topic X)
* Creates embeddings using Azure Open AI
* Creates an index an uploads results into Azure Cache for Redis
* Optional: Uploads results into Azure Cognitive Search
* Optional: Stores results into Cosmos DB
* Using Managed Identities where possible

## qna_redis.py

This script is an example of Q&A using Langchain with Azure Cache for Redis & Azure Open AI