from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# function to load the vectordatabase
def load_knowledge_base():
    embeddings = OpenAIEmbeddings()
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db


# function to load the OPENAI LLM
def load_llm():
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return llm


# creating prompt template using langchain
def load_prompt():
    prompt = """You are a helpful assistant that answers questions based on
        a vector database of San Francisco Airbnb listings.
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        If the answer is not in the pdf answer, "I do not know what the hell you are asking about"
        """
    prompt = ChatPromptTemplate.from_template(prompt)
    return prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
