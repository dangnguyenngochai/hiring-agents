import os
import sys
import embedding

from embedding import (
    api_loader,
    embedding_api_docs,
    EncodedApiDocVectorStore
)

import retrieval_generation
from retrieval_generation import (
    retrieval_text
)

from qdrant_client import QdrantClient

# from embeddings_api_docs import test_run
# from retrieval_generation import test_run2

from config import EMB_MODEL

def run(input, state, vstore_jd, vstore_res):
    response = retrieval_text.generate_response(input, state, vstore_jd, vstore_res)
    return response
    
if __name__ == "__main__":
    retrieval_text.test_run2()
    print("hello")