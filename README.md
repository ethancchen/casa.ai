# Casa.ai

Your home-finding companion tailoring properties to suit your needs. This chatbot is able to take in questions about the desired location, duration, budget, amenities, neighborhood, etc. then personalize those results accordingly using Retrieval-Augmented Generation (RAG).

## How does it function?

Describe what you are seeking in your ideal home to our virtual property concierge.
The model will then look at existing catalogs to help you find the most optimal option depending upon your needs!

## What impacts does it bring?

## Tech stack used

We used the following libraries:

- [OctoAI](https://octo.ai)
- [Streamlit](https://streamlit.io)
- [OpenAI](https://openai.com)
- [FAISS](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)
- [LangChain](https://www.langchain.com)

We used this [Kaggle dataset](https://www.kaggle.com/datasets/jeploretizo/san-francisco-airbnb-listings) to illustrate the capabilities of RAG in our demo. It contains 8000+ detailed Airbnb listings in San Francisco, CA.
