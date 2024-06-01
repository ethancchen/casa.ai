from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def processCSV(df):
    openai = OpenAIEmbeddings(openai_api_key="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")
    # Chunk data
    chunks = [df[i:i+openai.chunk_size] for i in range(0, df.shape[0], openai.chunk_size)]
    vector_store = FAISS()

    # Processing
    for chunk in chunks:
        text = chunk['summary'].tolist()
        embeddings = openai.embed_documents(text)
        vector_store.from_documents(text, embeddings)

    return vector_store