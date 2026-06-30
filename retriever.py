from langchain_community.vectorstores import FAISS
from embedding import EmbeddingModel
class Retriever:
    def __init__(self,embedding_model,query):
        self.embedding_model = embedding_model
        self.query = query
        self.vector_db = FAISS.load_local('VectorStore/faiss_index',
        self.embedding_model,
        allow_dangerous_deserialization=True)
    def retrieve_document(self):
        
        
        retriever = self.vector_db.as_retriever(
            search_type ='similarity',
            search_kwargs={'k':5}
        )
        
        result = retriever.invoke(self.query)
        context_list = [contxt.page_content for contxt in result]
        final_context = '\n\n'.join(context_list)
        return final_context

# model_name ='sentence-transformers/all-MiniLM-L6-v2'
# embedding= EmbeddingModel(model_name)
# embedding_model = embedding.load_model()
# retriever = Retriever(embedding_model,'what is neural network')
# context = retriever.retrieve_document()
# print(type(context[0]))
# print(context[0].page_content)
