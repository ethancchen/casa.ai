import streamlit as sl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


#function to load the vectordatabase
def load_knowledgeBase():
        embeddings=OpenAIEmbeddings()
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_llm():
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        return llm

#creating prompt template using langchain
def load_prompt():
        prompt = """You are a helpful assistant that answers the question based on the content of San Francisco Airbnb listings,
        which are currently stored as a vector database. 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        If the answer is not in the pdf answer, "I do not know what the hell you are asking about"
         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt


def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


if __name__=='__main__':
        sl.header("welcome to the 📝PDF bot")
        sl.write("🤖 You can chat by Entering your queries ")
        knowledgeBase=load_knowledgeBase()
        llm=load_llm()
        prompt=load_prompt()
        
        query=sl.text_input('Enter some text')
        
        
        if(query):
                #getting only the chunks that are similar to the query for llm to produce the output
                similar_embeddings=knowledgeBase.similarity_search(query)
                similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings())
                
                #creating the chain for integrating llm,prompt,stroutputparser
                retriever = similar_embeddings.as_retriever()
                rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                
                response=rag_chain.invoke(query)
                sl.write(response)