import sys
import os
import gradio as gr
import gradio_app
from gradio_app.theme import minigptlv_style, custom_css,text_css

from config import EMB_MODEL

from embedding_docs import EncodedDocVectorStore
import retrieval_text

from qdrant_client import QdrantClient

sys.path.append('/embedding')
sys.path.append('/retrieval_generation')

import argparse

DEMO_VSTORE_JD = None
DEMO_VSTORE_RE = None

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='test')  

def run(input, state, vstore_jd, vstore_res):
    response = retrieval_text.generate_response(input, state, vstore_jd, vstore_res)
    return response

def evaluate_candidate(ans, state):
    resp = run(ans, state, DEMO_VSTORE_JD, DEMO_VSTORE_RE)
    return resp

title = """<h1 Hiring Agent align="center"></h1>"""
description = """<h5>This is the demo for Hiring agent</h5>"""
with gr.Blocks(title="Hiring agent 🎞️🍿",css=text_css ) as demo :

    gr.Markdown(title)
    gr.Markdown(description)
        
    chatbot = gr.Chatbot(label="WebGPT")
    state = gr.State([])
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter")
    try:
        txt.submit(evaluate_candidate, [txt, state], [chatbot, state])
    except Exception as ex:
        print(ex)
       
if __name__ == "__main__":
    args = parser.parse_args()
    data_path = 'data'
        
    print("Setting things up....")
    
    _ = EMB_MODEL #load embedding model into memory

    # embedding api docs
    collection_name = 'job_description'
    
    qdrant_client = QdrantClient(location=":memory:")
    vstore_jd = EncodedDocVectorStore(collection_name=collection_name, qdrant_client=qdrant_client, model=EMB_MODEL)
    jd_path = os.path.join([data_path, 'jd']),

    for file in os.listdir(jd_path):
        path = os.path.join(jd_path, file)
        print(path)
        vstore_jd.embeddings_docs(path, collection_name)

    collection_name = 'resume'
    vstore_candidate = EncodedDocVectorStore(collection_name=collection_name, qdrant_client=qdrant_client, model=EMB_MODEL)
    resume_path = os.path.join([data_path, 'resume'])

    for file in os.listdir(resume_path):
        path = os.path.join(resume_path, file)
        print(path)
        vstore_candidate.embeddings_docs(path, collection_name)
        
    DEMO_VSTORE_JD= vstore_jd
    DEMO_VSTORE_RE = vstore_candidate

    print("Done")
    demo.queue().launch(share=True,show_error=True, server_port=2411)
