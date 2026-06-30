from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class LLMIntegration:
    def __init__ (self,context,query):
        self.query = query
        self.context = context
        self.model = ChatGoogleGenerativeAI(
            model ='gemini-2.5-flash'
        )
        self.parser = StrOutputParser()
        
        self.prompt = ChatPromptTemplate.from_messages(
                  [
                                ('system',
                            '''You are a helpful AI assistant.

                        Answer the user's question only using the provided context.

                        If the answer is not present in the context, say:
                        "I don't know based on the provided document."

                        Be concise, accurate, and do not make up information.'''),
                            (
                                'human',
                                '''
                                context :{context}
                                 question:{query}
                                '''
                            )
                  ])
 
    def generate_response(self):
         chain = self.prompt|self.model|self.parser
         response = chain.invoke(
             {
                 'context':self.context,
                 'query':self.query
             }
         )
         return response
# context ='bining a deep knowledge of network fundamentals with practical experi\xad\nence in using neural networks that we can get the most out of this \ntechnology.\nFigure 22.1 illustrates the neural network training process. It is an itera\xad\ntive procedure that begins by collecting data and preprocessing it to make \ntraining more efficient. At this stage, the data also needs to be divided into \ntraining/validation/testing sets (see Chapter 13). After the data is selected,'
# llm = LLMIntegration(context,'what is context about')
# response = llm.generate_response()
# print(response)
