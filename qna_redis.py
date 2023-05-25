import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores.redis import Redis
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI

load_dotenv()
openai_endpoint = os.environ["OPENAI_ENDPOINT"]
openai_embeddings = os.environ["OPENAI_EMBEDDINGS"]
openai_completions = os.environ["OPENAI_COMPLETIONS"]
redis_endpoint = os.environ["REDIS_ENDPOINT"]
redis_key = os.environ["REDIS_KEY"]
redis_index = os.environ["REDIS_INDEX"]
redis_port = os.environ["REDIS_PORT"]

redis_url = f"rediss://:{redis_key}@{redis_endpoint}:{redis_port}"

credential = DefaultAzureCredential()
oai_token = credential.get_token("https://cognitiveservices.azure.com/.default")

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, say that you don't know, don't try to make up an answer.

    This should be in the following format:

    Question: [question here]
    Answer: [answer here]

    Begin!

    Context:
    ---------
    {context}
    ---------
    Question: {question}
    Answer:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

embeddings = OpenAIEmbeddings(
    openai_api_type="azure",
    openai_api_base=openai_endpoint,
    deployment=openai_embeddings,
    #openai_api_key=oai_token.token
)

redis = Redis.from_existing_index(
    embedding=embeddings,
    index_name=redis_index,
    redis_url=redis_url
)

llm = AzureOpenAI(
    # openai_api_type="azure_ad",
    openai_api_base=openai_endpoint,
    deployment_name=openai_completions,
    # openai_api_key=oai_token.token
)

# Create retreival QnA Chain
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=redis.as_retriever(),
    # return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

query = "Ask a question about your data here"
result = chain.run(query)

print(result)

