from langchain_community.vectorstores import FAISS
# from embedding import EmbeddingModel

# embedding = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2")
# embedding_model = embedding.load_model()
class Query:
    def __init__(self,embedding_model,query):
      self.query = query
      self.embedding_model = embedding_model
      self.vector_db = FAISS.load_local(
           "VectorStore/faiss_index",
             embedding_model,
             allow_dangerous_deserialization=True
                  )
    def get_result(self):
      result = self.vector_db.similarity_search(self.query, k=1)
      return result
