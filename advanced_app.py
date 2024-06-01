import streamlit as sl
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from basic_app import OCTO_API
import requests, base64


# function to load the vectordatabase
def load_knowledgeBase():
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

# Function for converting img
def encoded_img():
    img = requests.get(
        "https://a0.muscache.com/im/pictures/29e0cc65-47dc-41ad-a279-978152ab2899.jpg?im_w=720").content
    with open(img, "wb") as f:
        encoded_img = base64.b64encode(f)
    return encoded_img


if __name__ == "__main__":
    sl.header("welcome to the 📝PDF bot")
    sl.write("🤖 You can chat by Entering your queries ")
    knowledgeBase = load_knowledgeBase()
    llm = load_llm()
    prompt = load_prompt()

    query = sl.text_input("Enter some text")

    if query:
        # getting only the chunks that are similar to the query for llm to produce the output
        similar_embeddings = knowledgeBase.similarity_search(query)
        similar_embeddings = FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings())

        # creating the chain for integrating llm,prompt,stroutputparser
        retriever = similar_embeddings.as_retriever()
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
        )

        response = rag_chain.invoke(query)
        sl.write(response)

        video_gen_response = OCTO_API.image_gen.generate_svd(
            image=encoded_img(),
            cfg_scale=5,
            steps=20,
            motion_scale=0.5,
            noise_aug_strength=0.04,
            num_videos=1,
            fps=30
        )
        sl.write(video_gen_response.videos)
