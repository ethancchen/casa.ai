import base64
import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from octoai.client import OctoAI

load_dotenv()


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


def generate_video():
    with open("v2.png", "rb") as picture:
        picture_data = picture.read()

        # Encode the file content to base64
        base64_string = base64.b64encode(picture_data)

        # Convert the bytes to string
        base64_string = base64_string.decode("utf-8")

    client = OctoAI(
        api_key=os.environ["OCTOAI_API_TOKEN"],
    )
    response = client.image_gen.generate_svd(
        image=base64_string, cfg_scale=3, steps=25, motion_scale=0.5, noise_aug_strength=0.02, num_videos=1, fps=7
    )
    # import pdb; pdb.set_trace()
    print(response.videos)
    video_base64 = response.videos[0].video  # Adjust this based on the actual response structure

    # Decode the base64 video data
    video_data = base64.b64decode(video_base64)

    # Save the video data to a file
    with open("generated_video.mp4", "wb") as video_file:
        video_file.write(video_data)

    print("Video saved as 'generated_video.mp4'")
