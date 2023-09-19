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
from langchain.schema import HumanMessage
from langchain.cache import InMemoryCache


from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, StringPromptTemplate


os.environ["OPENAI_API_KEY"] = ""
# openai.api_key = os.environ.get("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key = os.environ.get("OPENAI_API_KEY"),model_name="text-embedding-ada-002")

persist_directory = 'C:/Users/arman/Workspace/phd/Arman/autostances/data/chroma_database/'
db_client = chromadb.PersistentClient(path=persist_directory)

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.7)

langchain.llm_cache = InMemoryCache()

print("Here are your available collections: ", db_client.list_collections())

class RedditPromptTemplate(StringPromptTemplate):

  def format(self, **kwargs) -> str:
    context_query = """Use the following two sources of context to complete the task. There are two sources, Reddit and U.S Congressional Hearings. For Reddit context there will be 5 lists, and for Congressional hearings there will be 8 lists. Each position in the list corresponds to metadata for a single post or single speaker. Use both contexts equally. For Reddit use the Comments as the source of context, for Congressional hearings use Speech as the source of context. Retain all metadata information for postprocessing. \n
    
    Authors: {author} \n
    Comments: {comment} \n
    score: {score} \n
    forum: {forum} \n
    title: {title} \n\n

  The second context is from United States Congressional Hearings. The structure is as follows: \n
  Congress:{congress}\n
  Chamber:{chamber}\n
  Committee Name: {committee_name}\n
  Title:{title}\n
  Ranking:{ranking}\n
  First Name: {speaker_first}\n
  Last Name: {speaker_last}\n
  Speech: {speech2}\n\n

    
    Task: Create an argument diagram, with at least two arguments for AND against the topic {topic} using the previous contexts. 
    Here are instructions for completing the task. Follow the structure to create the exact output.\n

    # An argument diagram consists of the following steps:
    # - Provide the argument for or against the topic as a new entry in a numbered list.
    # - Elaborate on the argument as to why it is for or against the topic as a bullet point.
    # - Provide a strong counterargument against the argument above as a bullet point.
    # - If another counterargument exists against the original argument, provide the counterargument. Repeat until no more counterarguments exist.
    # - Provide a strong refutation against the counterargument, and in support of the original argument.
    # - If more strong refutations exist against the counterargument, provide them. Repeat this step until no more strong refutations exist.\n

    # - For each entry in the numbered list and for each bullet point, include if the reference came from relevant Reddit comments context provided or Congressional hearings Speech. If the entry came from Reddit, credit the author of the comment and the forum it was posted in. If the entry came from Congressional Hearings credit the congressional title and speaker name. If the knowledge did not come from the context, say so.\n
    Extra Instructions:\n
    # - Supplement the argument diagram with general knowledge.\n
    # - After providing the argument diagram, provide a list of topics that are commonly used in arguments made about {topic}, using both of the contexts provided and general knowledge. For each concept or topic cite whether it is from the Reddit context (author and/or forum), Congressional Hearing (Title and Speaker name) or general knowledge.
  

    Complete the task and the output should be as long as necessary:
    """

    doc = kwargs['reddit_context']['metadatas'][0][0]
    kwargs['author'] = doc.get('author')
    kwargs['comment'] = doc.get('comment')
    # kwargs['comment_length'] = doc.get('comment_length')
    # kwargs['id'] = doc.get('id')
    # kwargs['parent_id'] = doc.get('parent_id')
    # kwargs['is_op'] = doc.get('is_op')
    kwargs['score'] = doc.get('score')
    kwargs['forum'] = doc.get('forum')
    kwargs['title'] = doc.get('title')
    kwargs["topic"] = topic

    doc2 = kwargs['congress_context']['metadatas'][0][0]
    kwargs['congress'] = doc2.get('congress')
    kwargs['chamber'] = doc2.get('chamber')
    kwargs['committee_name'] = doc2.get('committee_name')
    # kwargs['committee_code'] = doc2.get('committee_code')
    kwargs['title'] = doc2.get('title')
    # kwargs['govtrack'] = doc2.get('govtrack')
    kwargs['ranking'] = doc2.get('ranking')
    kwargs['speaker_first'] = doc2.get('speaker_first')
    kwargs['speaker_last'] = doc2.get('speaker_last')
    kwargs['speech2'] = doc2.get('speech2')
    # kwargs["topic"] = topic
    return context_query.format(**kwargs)
  
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


def multiquery():
  try:
    reddit_collection = db_client.get_collection("reddit")
  except:
    print("No collection found")
  langchain_chroma = Chroma(
    client=db_client,
    collection_name=reddit_collection.name,
    embedding_function=OpenAIEmbeddings(model='text-embedding-ada-002')).as_retriever(search_type="mmr")
  
  llm = ChatOpenAI(model_name='gpt-4',temperature=0.5)

  question = "What are the arguments for and against GMOs?"
 
  QUERY_PROMPT = PromptTemplate(
    input_variables=["topic"],
    template="""
  Use the following vectorstore of Reddit comments as context: \n
    
    Task: Create an argument diagram, with at least two arguments for AND against the topic {topic} using the previous contexts. 
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
  
  """,
  )
  topic = "GMOS"
  QUERY_PROMPT.format(topic=topic)
  llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT)
  # llm_chain.run({"topic": topic})
  # llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, input_variables={"topic": topic})
  
  retriever = MultiQueryRetriever(
    retriever=langchain_chroma, llm_chain=llm_chain,input_key=topic)  
  
  # retriever
  docs = retriever.get_relevant_documents(query="GMOs")
  print(docs)

# multiquery()
# query_reddit_v2()



def retrieve_reddit(topic):
  try:
    reddit_collection = db_client.get_collection("reddit")
  except:
    print("No collection found")
  retrieve_relevant_docs = "Retrieve all data that contain information pertaining to " + topic + " or concepts relating to " + topic + "."

  res = openai.Embedding.create(input=[retrieve_relevant_docs],
                                engine='text-embedding-ada-002')

  # retrieve from Pinecone
  xq = res['data'][0]['embedding']
  # print(xq)

  # get relevant contexts\
  res = reddit_collection.query(query_embeddings=xq,
                    n_results=25,)
  print(res)
  # contexts = [x['metadata']['comment'] for x in res['matches']]

  return res


def retrieve_congress(topic):
  try:
    congress_collection = db_client.get_collection("congress")
  except:
    print("No collection found")
  retrieve_relevant_docs = "Retrieve all data that contain information pertaining to " + topic + " or concepts relating to " + topic + "."
  res = openai.Embedding.create(input=[retrieve_relevant_docs],
                                engine='text-embedding-ada-002')

  # retrieve from Pinecone
  xq = res['data'][0]['embedding']

  # get relevant contexts
  res = congress_collection.query(query_embeddings=xq,
                    n_results=25)
  # contexts = [x['metadata']['comment'] for x in res['matches']]

  return res


def complete(topic, reddit_context, congress_context):

  prompt_template = RedditPromptTemplate(
    input_variables=["reddit_context", "congress_context", "topic"])
  return llm([
    HumanMessage(
      content=prompt_template.format(reddit_context=reddit_context,
                                     congress_context=congress_context,
                                     topic=topic))
  ]).content

topic = 'GMO'



reddit_context = retrieve_reddit(topic)
congress_context = retrieve_congress(topic)

# print(query_with_contexts)
print(complete(topic, reddit_context, congress_context))