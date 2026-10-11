from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import chromadb

db = chromadb.PersistentClient('./travel_db')



# load env
load_dotenv()

# init API_KEY for OpenAI API
api_key = os.getenv('API_KEY')

# file laoders
attractions_loader = UnstructuredMarkdownLoader('data/attractions.md')
transport_loader = UnstructuredMarkdownLoader('data/transport.md')
travel_tips_loader = UnstructuredMarkdownLoader('data/travel_tips.md')

# load file content
attractions_data = attractions_loader.load()
transport_data = transport_loader.load()
travel_tips_data = travel_tips_loader.load()

# splitting documents into chunks
rc_splitter = RecursiveCharacterTextSplitter(
    separators=['.'],
    chunk_size=500,
    chunk_overlap=100
)

# merge all docs
all_docs = rc_splitter.split_documents(attractions_data + transport_data + travel_tips_data)

# # init OpenAI embedding model function
# embedding_function = OpenAIEmbeddings(api_key=api_key, model='text-embedding-3-small')

# # init collection create logic
# vectorstore = Chroma.from_documents(
#     documents=all_docs,
#     embedding=embedding_function,
#     persist_directory='./travel_db',
#     collection_name='budapest_attractions'
# )


