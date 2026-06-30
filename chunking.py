from langchain_text_splitters import RecursiveCharacterTextSplitter

class GenerateChunks:
    def __init__(self,documents,chunk_size,chunk_overlap):
        
        self.documents = documents
        self.chunk_size= chunk_size
        self.chunk_overlap = chunk_overlap
    def chunking(self):
        try:
            print(f'generating chunk...')
            text_splitter= RecursiveCharacterTextSplitter(
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap )
            chunk = text_splitter.split_documents(self.documents)
            return chunk
        except FileNotFoundError:
            raise FileNotFoundError(f'File not found {self.documents}\n')
        except Exception as e:
            raise RuntimeError(f'Failed to chunk documents{e}.\n')
        
            

