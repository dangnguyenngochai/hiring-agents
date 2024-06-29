
from retrieval_generation import (
    retrieval_text
)
def run(input, state, vstore_jd, vstore_res):
    response = retrieval_text.generate_response(input, state, vstore_jd, vstore_res)
    return response
    
if __name__ == "__main__":
    retrieval_text.test_run2()
    print("hello")