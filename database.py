# import chromadb
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import preprocessing as preprocessing
import os
import langchain
import openai
import sys
from time import sleep
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import VectorDBQA, RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

from chromadb.utils import embedding_functions
import chromadb 
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


os.environ["OPENAI_API_KEY"] = ""
# openai.api_key = os.environ.get("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key = os.environ.get("OPENAI_API_KEY"),model_name="text-embedding-ada-002")

persist_directory = 'C:/Users/arman/Workspace/phd/Arman/autostances/data/chroma_database/'
db_client = chromadb.PersistentClient(path=persist_directory)


print("Here are your available collections: ", db_client.list_collections())

  
def query_reddit_v2():
  prompt= """
  Use the following vectorstore of Reddit comments as context: \n
    
    Task: Create an argument diagram, with at least two arguments for AND against the topic GMOs using the previous contexts. 
    Here are instructions for completing the task. Follow the structure to create the exact output.\n

    # An argument diagram consists of the following steps:
    # - Provide the argument for or against the topic as a new entry in a numbered list.
    # - Elaborate on the argument as to why it is for or against the topic as a bullet point.
    # - Provide a strong counterargument against the argument above as a bullet point.
    # - If another counterargument exists against the original argument, provide the counterargument. Repeat until no more counterarguments exist.
    # - Provide a strong refutation against the counterargument, and in support of the original argument.
    # - If more strong refutations exist against the counterargument, provide them. Repeat this step until no more strong refutations exist.\n

    # - For each entry in the numbered list and for each bullet point, include if the reference came from relevant Reddit comments context provided. If the entry came from Reddit, credit the author of the comment and the forum it was posted in. If the knowledge did not come from the context, say so.\n
    Extra Instructions:\n
    # - Supplement the argument diagram with general knowledge.\n
    # - After providing the argument diagram, provide a list of topics that are commonly used in arguments made about GMOs, using both of the contexts provided and general knowledge. For each concept or topic cite whether it is from the Reddit context (author and/or forum), or general knowledge.
  

    Complete the task and the output should be as long as necessary:
  
  """
  try:
    reddit_collection = db_client.get_collection("reddit")
  except:
    print("No collection found")
  langchain_chroma = Chroma(
    client=db_client,
    collection_name=reddit_collection.name,
    embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002')).as_retriever(search_type="mmr")
  
  llm = ChatOpenAI(model_name='gpt-4',temperature=0.5)

  embeddings_filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(model='text-embedding-ada-002'), similarity_threshold=0.76)
  compression_retriever = ContextualCompressionRetriever
  compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=langchain_chroma)
  compressed_docs = compression_retriever.get_relevant_documents("GMOs")

  print(compressed_docs)
query_reddit_v2()



