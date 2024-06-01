import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from octoai.client import OctoAI
from openai import OpenAI

from advanced_app import encoded_img
from utils import format_docs, load_knowledgeBase, load_llm, load_prompt

load_dotenv()
client = OpenAI()
OCTO_API_KEY = os.getenv("OCTO_API_KEY")

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
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)
            video_gen_response = OctoAI(OCTO_API_KEY).image_gen.generate_svd(
                image=encoded_img(),
                cfg_scale=5,
                steps=20,
                motion_scale=0.5,
                noise_aug_strength=0.04,
                num_videos=1,
                fps=30,
            )
            st.write(video_gen_response.videos)
