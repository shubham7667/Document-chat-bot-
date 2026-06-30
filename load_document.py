from langchain_community.document_loaders import PyMuPDFLoader
class DocumentLoader:
    def __init__(self,file_path):
        self.file_path = file_path
        
    def load_docs(self):
        try:
          loader = PyMuPDFLoader(self.file_path)
          document = loader.load()
          return document
        except FileNotFoundError:
            raise FileNotFoundError(f'File not found{self.file_path}')
        except Exception as e:
            raise RuntimeError(f'fail to load document{e}')
     
# path = 'Data/NNDesign.pdf'
# loader = DocumentLoader(path)
# Data = loader.load_docs()

# print(Data[30].page_content)