from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_cohere import CohereEmbeddings
import os

os.environ['COHERE_API_KEY'] = 'OhqrR0Bude8zr30XWVjreKC4LNbNHPhivxw7n0Vw'

print("Downloading embedding")

# HG_EMB_MODEL = HuggingFaceEmbeddings(
#                 model_name="Alibaba-NLP/gte-large-en-v1.5",
#                 model_kwargs={
#                     'trust_remote_code': True
#                 }
#             )
COHERE_EMB_MODEL = CohereEmbeddings(model="embed-english-light-v3.0", )
