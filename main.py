from load_document import DocumentLoader
from chunking import GenerateChunks
from embedding import EmbeddingModel
from vector_db import VectorStore
from query import Query
from retriever import Retriever
from llm import LLMIntegration

# Load document
loader = DocumentLoader('Data/NNDesign.pdf')
document = loader.load_docs()

# Generate chunks
get_chunk = GenerateChunks(document,500,100)
chunks = get_chunk.chunking()


# Generate Embeddings
model_name ='sentence-transformers/all-MiniLM-L6-v2'
embedding =EmbeddingModel(model_name)
embedding_model=embedding.load_model()


# Vector Store
database_manager = VectorStore(chunks,embedding_model)
vector_db = database_manager.create_vector_db() 

# query search in DB
query_input = input('Enter you query?')
# get_query = Query(embedding_model,query_input)
# result = get_query.get_result()
# print(result,'\n',result)

# retriever
retriever = Retriever(embedding_model,query_input)
context = retriever.retrieve_document()



# giving context + query to llm
llm = LLMIntegration(context,query_input)
response = llm.generate_response()
print(response)