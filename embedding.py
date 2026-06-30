from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    def __init__(self,model_name):
        self.model_name = model_name
  
    def load_model(self):
        try:
         print('generating embeddings')
         model = HuggingFaceEmbeddings(
            model =self.model_name
         )
         return model
        except Exception as e:
            raise RuntimeError(f'Failed to generate embedding{e}')
        
