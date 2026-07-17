import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == "__main__":
    loader = TextLoader(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mediumblog1.txt"), encoding="utf-8")
    document = loader.load()

    print("Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")


    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("DEPLOYMENT"),
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    )

    # print("Embedding texts...")
    # for i, text in enumerate(texts):
    #     vector = embeddings.embed_query(text.page_content)
    #     print(f"\n--- Chunk {i + 1} ---")
    #     print(f"Content: {text.page_content[:200]}...")
    #     print(f"Vector length: {len(vector)}")
    #     print(f"First 5 values: {vector[:5]}")
    # print("\nDone!")

    print("Ingesting...")

    # SSL error
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    print(f"created {len(texts)} Pinecone vectors")
