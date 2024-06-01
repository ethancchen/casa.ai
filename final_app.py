import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from utils import format_docs, load_knowledgeBase, load_llm, load_prompt

load_dotenv()
client = OpenAI()

st.title("casa.ai")

if __name__ == "__main__":
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "knowledge_base" not in st.session_state:
        st.session_state["knowledge_base"] = load_knowledgeBase()

    if "llm" not in st.session_state:
        st.session_state["llm"] = load_llm()

    if "prompt" not in st.session_state:
        st.session_state["prompt"] = load_prompt()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            similar_embeddings = st.session_state["knowledge_base"].similarity_search(prompt)
            similar_embeddings = FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings())

            # creating the chain for integrating llm,prompt,stroutputparser
            retriever = similar_embeddings.as_retriever()
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | st.session_state["prompt"]
                | st.session_state["llm"]
                | StrOutputParser()
            )

            response = rag_chain.invoke(prompt)
            # stream = client.chat.completions.create(
            #     model=st.session_state["openai_model"],
            #     messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
            #     stream=True,
            # )
            response = st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
