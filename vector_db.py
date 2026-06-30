from langchain_community.vectorstores import FAISS

class VectorStore:
    def __init__(self,chunks,embedding_model):
        self.chunks = chunks
        self.embedding_model = embedding_model
    
    def create_vector_db(self):
        vector_db =FAISS.from_documents(
            self.chunks,
            self.embedding_model
        )
        # vector_db.save_local('VectorStore/faiss_index')
        return vector_db
    

    
