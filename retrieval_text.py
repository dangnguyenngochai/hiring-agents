import os
from embedding_docs import EncodedDocVectorStore, test_run as dummy_emb

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_google_genai import (
#     ChatGoogleGenerativeAI
# )
from langchain import hub
from langchain_cohere import ChatCohere

import langchain
langchain.debug=True

# os.environ["OPENAI_API_KEY"] = 'sk-proj-0VuYdXfFXk6evfzOzLJTT3BlbkFJcshpzLyIT1ij7tR8q11Q'
# os.environ["GOOGLE_API_KEY"] = 'AIzaSyB-PQLnrQm2Z5UGXW28R24TXG99MLPICFw'

def prompt_generator(mode):
    if mode=='questioner':
        sys_message = """
        SYSTEM:
        You are an interviewer for a job position with description in the JOB DESCRIPTION section \
        and the candidate has the resume in the RESUME section.
        Using information provided by the evaluator in EVALUATOR ANALYSIS on the their latest answers, \
        you need to come up with the follow up questions in the following topics: \
        
        technical question behavioral questions, \
        and questions related to the candidate's past experience using the candidate RESUME.

        JOB DESCRIPTION:
        {jd}
        RESUME:
        {re}
        EVALUATOR ANALYSIS:
        {eval}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system",sys_message),
            ("ai", "Please answer the following question"),
            ("human", "Yes")
        ])
        return prompt
    
    elif mode=='evaluator':
        sys_message = """
        You are an seninor level professionals in the field mentioned in the JOB DESCRIPTION section.
        Given your expertise, you need to evaluate a candidate knowledge base on the CANDIDATE ANSWER according to INTERVIEWER QUESTION
        As the evaluator, you need to check if the answer reflect their experience and their skills,
        which are described the RESUME section, and providing analysis on their answer: \
        How relevance the answer is to the question ? How correct the answer is ? and How complete the answer is ?

        JOB DESCRIPTION:
        {jd}
        RESUME:
        {re}
        INTERVIEWER QUESTION:
        {iq}
        CANDIDATE ANSWER:
        {ca}
        """
        prompt = PromptTemplate.from_template(sys_message)
        return prompt
        
        
def generate_response(input: str, chat_history: list, vstore_jd: EncodedDocVectorStore, vstore_res: EncodedDocVectorStore): 
    def format_docs(docs):
        output =  "\n\n".join(doc.page_content for doc in docs)
        print(output)
        return output

    # for simplicity, get the last question only
    def get_interviewer_question(chat_history):
        last_topic = chat_history[-1]
        return last_topic[-1]

    # fetch prompt template
    # prompt = hub.pull("rlm/rag-prompt")

    # eval
    mode = 'evaluator'
    prompt = prompt_generator(mode)
    retriever_jd = vstore_jd.get_retriever(5) # retrieve with meta data
    retriever_re = vstore_res.get_retriever(5) # retrieve with meta data
    
    llm = ChatCohere(model="command-r-plus")

    rag_chain = (
        {"jd": itemgetter('input') | retriever_jd | format_docs,
         "re": itemgetter('input') | retriever_re | format_docs,
         "iq": itemgetter('state'),
         "ca": itemgetter('input')}
        | prompt
        | llm   
        | StrOutputParser()
    )
    
    eval_response = rag_chain.invoke({'state': get_interviewer_question(chat_history), 'input': input})

    print(eval_response)

    mode = 'questioner'
    prompt = prompt_generator(mode)

    llm = ChatCohere(model="command-r")

    summary_chain = (
        {"jd": retriever_jd | format_docs, 
         "re": retriever_re | format_docs,
         "eval": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    questioner_response = summary_chain.invoke(eval_response)

    return questioner_response

def test_run2():
    query = "Which is the api for listing the ads account?"
    dummy_vt = dummy_emb(run_query=False)
    response = generate_response(query, dummy_vt)
    print(response)
    
def test_run3():
    query = "Campaign"
    response = generate_response(query)
    print(response)
    
if __name__ == '__main__':
    test_run2()