import base64

import requests
import streamlit as sl
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

from basic_app import OCTO_API
from utils import format_docs, load_knowledgeBase, load_llm, load_prompt


# Function for converting img
def encoded_img():
    img = requests.get("https://a0.muscache.com/im/pictures/29e0cc65-47dc-41ad-a279-978152ab2899.jpg?im_w=720").content
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
            image=encoded_img(), cfg_scale=5, steps=20, motion_scale=0.5, noise_aug_strength=0.04, num_videos=1, fps=30
        )
        sl.write(video_gen_response.videos)
