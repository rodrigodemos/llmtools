import os
import re
import time
import datetime
import string
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.search.documents import SearchClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter


load_dotenv()
fr_endpoint = os.environ["FORM_RECOGNIZER_ENDPOINT"]
search_endpoint = os.environ["SEARCH_ENDPOINT"]
index_name = os.environ["SEARCH_INDEX"]
cosmosdb_endpoint = os.environ["COSMOSDB_ENDPOINT"]
cosmosdb = os.environ["COSMOSDB_DATABASE"]
cosmosdb_container = os.environ["COSMOSDB_CONTAINER"]
openai_endpoint = os.environ["OPENAI_ENDPOINT"]
openai_embeddings = os.environ["OPENAI_EMBEDDINGS"]
redis_endpoint = os.environ["REDIS_ENDPOINT"]
redis_key = os.environ["REDIS_KEY"]
redis_index = os.environ["REDIS_INDEX"]
redis_port = os.environ["REDIS_PORT"]

redis_url = f"rediss://:{redis_key}@{redis_endpoint}:{redis_port}"

file = "C:/test/file.pdf"

credential = DefaultAzureCredential()
oai_token = credential.get_token("https://cognitiveservices.azure.com/.default")

def add_to_index(search_client, docs):
    i = 0
    batch = []
    for doc in docs:
        batch.append(doc)
        i += 1
        if i % 1000 == 0:
            index_results = search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in index_results if r.succeeded])
            batch = []
    if len(batch) > 0:
        results = search_client.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])

    return succeeded

def clean_text(text):
    if not text:
        return ""
    # remove unicode characters
    text = text.encode('ascii', 'ignore').decode()

    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    # clean up the spacing
    text = re.sub('\s{2,}', " ", text)

    # remove urls
    #text = re.sub("https*\S+", " ", text)

    # remove newlines
    text = text.replace("\n", " ")

    # remove all numbers
    #text = re.sub('\w*\d+\w*', '', text)

    # split on capitalized words
    #text = " ".join(re.split('(?=[A-Z])', text))

    # clean up the spacing again
    text = re.sub('\s{2,}', " ", text)

    # make all words lowercase
    text = text.lower()

    return text.strip()

def create_embeddings(embeddings, docs):
    throttle = 0
    start_time =datetime.datetime.now()
    for doc in docs:
        clean_content = clean_text(doc["content"])
        doc["embedding"] = embeddings.embed_query(clean_content) if clean_content else None
        throttle += 1
        if throttle >= 300:
            current_time = datetime.datetime.now()
            if (current_time - start_time).seconds < 60:
                time.sleep(60 - (current_time - start_time).seconds)
                throttle = 0
            
def shape_redis_docs(docs):
    redis_docs = [
        Document(
            page_content=doc["content"],
            metadata={
                "id": doc["id"],
                "form": doc["form"],
                "file": doc["file"],
                "shorturl": doc["shorturl"],
                "page": doc["page"],
                "section": doc["section"],
                "subsection": doc["subsection"],
                "topic": doc["topic"]
            }
        ) for doc in docs
    ]

    return redis_docs

def add_to_vectorstore(embeddings, docs):
    for doc in docs:
        doc['content'] = clean_text(doc['content'])

    redis_docs = shape_redis_docs(docs)
    for doc in redis_docs:
        vectorstore = Redis.from_documents(
            documents=[doc],
            embedding=embeddings,
            index_name=redis_index,
            redis_url=redis_url
        )
    return vectorstore

document_analysis_client = DocumentAnalysisClient(endpoint=fr_endpoint, credential=credential)

search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)
embeddings = OpenAIEmbeddings(
        openai_api_type="azure_ad",
        openai_api_base=openai_endpoint,
        deployment=openai_embeddings,
        openai_api_key=oai_token.token
        )

# cosmos_client = CosmosClient(url=cosmosdb_endpoint, credential=credential)
# cosmos_db = cosmos_client.get_database_client(cosmosdb)
# cosmos_cont = cosmos_db.get_container_client(cosmosdb_container)

with open(file, "rb") as f:
    poller = document_analysis_client.begin_analyze_document("prebuilt-layout", document = f)

fr_result = poller.result()

#find first ocurrence of role == "pageFooter" in result.paragraphs
form = [paragraph.content for paragraph in fr_result.paragraphs if paragraph.role == "pageFooter"][0]

currentSection = None
clearSectionOnNext = False
filename = file.split("/")[-1]
fulldoc = fr_result.content
docs = []
paragraphs = []

for index, paragraph in enumerate(fr_result.paragraphs):
    page = paragraph.bounding_regions[0].page_number
    id = filename.replace(".","_") + "-" + str(page) + "-" + str(index)

    # TODO Ignore paragraph if paragraph.role == "pageFooter"

    if paragraph.role == "sectionHeading":
        if clearSectionOnNext:
            currentSection = paragraph.content
        else:
            # Append adjacent headings
            currentSection += paragraph.content
    else:
        clearSectionOnNext = True
        topic, *content = paragraph.content.split(":", 1)
        if not content:
            content = topic
            topic = None
        else:
            content = ":".join(content).strip()
        paragraph_dict = {
            "id": id,
            "form": form,
            "file": file,
            "shorturl": "https://aka.ms/fabric",
            "page": page,
            "section": currentSection,
            "subsection": "",
            "topic": topic,
            "content": content
        }
        #doc_result = cosmos_cont.upsert_item(body=paragraph_dict)
        docs.append(paragraph_dict)
        topic = None
        content = None
    paragraphs.append(paragraph.content)

#create_embeddings(embeddings, docs)
#add_to_index(search_client, docs)
redis = add_to_vectorstore(embeddings, docs)