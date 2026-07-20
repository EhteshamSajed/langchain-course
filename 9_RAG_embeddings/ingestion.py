import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings


load_dotenv()
print("USE_OLLAMA?", os.getenv("USE_OLLAMA"))
if __name__ == "__main__":
    loader = TextLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediumblog1.txt"), encoding="utf-8")
    document = loader.load()

    print("Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")


    if os.getenv("USE_OLLAMA", "false") == "true":
        embeddings = OllamaEmbeddings(model="bge-m3:latest")
    else:
        embeddings = AzureOpenAIEmbeddings(
            deployment=os.getenv("DEPLOYMENT"),
            azure_ad_token_provider=get_bearer_token_provider(
                DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        )

    e = embeddings.embed_query("test")  # test embedding to ensure it works

    print("Ingesting...")

    # SSL error
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    print(f"created {len(texts)} Pinecone vectors")
