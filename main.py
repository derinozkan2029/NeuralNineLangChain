import uuid
import os
import subprocess
from typing import TypedDict, Annotated, Literal 
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  #this is for intelligent embeddings
from langchain.chat_models import init_chat_model # importing this for the LLM model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver #for adding memory to the model
from langgraph.types import interrupt,Command 

load_dotenv()

llm =init_chat_model('openai:gpt-4.1-mini')


KNOWLEDGE = [
    "NeuralNine is a YouTube channel focused on programming, AI, and software engineering tutorials.",
    "LangGraph is a library for building stateful, multi-agent applications on top of LangChain.",
    "A StateGraph in LangGraph defines nodes and edges that operate on a shared typed state.",
    "Checkpointers like InMemorySaver let LangGraph persist conversation state across invocations using a thread_id.",
    "RAG (Retrieval-Augmented Generation) combines a retriever over a knowledge base with an LLM to ground answers in source documents.",
]

vector_store= InMemoryVectorStore(OpenAIEmbeddings(model= 'text-embedding-3-small'))
vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])

class IntentClassifier(BaseModel):
    message_intent: Literal['chat','knowledge', 'code'] =Field(..., description= 'Classify whether the user' \
    'just wants to chat, ask for knowledge or change code in the project.') #this is for determining which node to go to next

class State(TypedDict): #our own custom state inheriting from TypedDict
    messages: Annotated[list, add_messages]
    message_intent:str | None #this also needs to be passed down from node to node
    next_node  = str | None

def classify_intent(state:State): #takes a state and does an LLMP prompt with a structured output
    structured_llm = llm.with_structured_output(IntentClassifier)

    result = structured_llm.invoke([{'role':'system', 'content': 'Determine/classify whether the user wants to chat ("chat) , retrieve a knowledge'
    '("knowledge) or change code ("code)'}, 
    {'role': 'user', 'content': state['messages'][-1].content}]) #getting the state and the last message 
    #this creates an instance of an intent classifier 
    return {'message_intent': result.message_intent}

def accept_coding(state:State):
    user_prompt= state['messages'][-1].content
    decision= interrupt(f'About to run Claude Code with request:\n\n{user_prompt}\n\n Approve?(yes/no, or type a revised request)')
    text = str(decision).strip().lower()

    if text in ['y', 'yes', 'approve', 'ok']:
        return {'next_node': 'coding_agent'}

    if text in ['n', 'no', 'deny', 'cancel']:
        return {'messages': [{'role': 'assistant', 'content':'Coding request was denied by the user'}],'next_node':'denied'}

    return {'messages': [{'role': 'assistant', 'content': text}], 'next_node': 'accept_coding'} #this is for human in the loop


def prepare_coding_request(state: State):
    messages = [
        {'role': 'system', 'content': 'Rewrite the latest user coding request into a clear instruction for Claude Code. Use the conversation history as context. Only output the instruction, no explanation.'},
    ] + state['messages']

    response = llm.invoke(messages)

    return {'messages': [{'role': 'user', 'content': response.content}]}

def prompt_llm_chat(state:State):
    messages = [{'role': 'system', 'content': 'You are a talkative chatbot for fun. Be nice.'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]} #bc we are appending we pass this as a list
    # this one is just for chatting

def prompt_llm_rag(state:State):
    query = state['messages'][-1].content
    documents = vector_store.similarity_search(query,k=3)
    context= '\n'.join(f'- {doc.page_content}' for doc in documents)

    messages = [{'role': 'system', 'content': 'You are a an RAG agent.'}] + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]} #bc we are appending we pass this as a list

def prompt_llm_code(state:State):
    user_prompt = state['messages'][-1].content
    workspace= os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace')
    result = subprocess.run(['claude', '-p', user_prompt, '--permission-mode', 'acceptEdits'], cwd=workspace, capture_output=True, text=True)
    output= result.stdout.strip() or result.stderr.strip()
    
    return {'messages': [{'role': 'assistant', 'content':output}]} #bc we are appending we pass this as a list

graph_builder = StateGraph(State)
graph_builder.add_node('classifier', classify_intent)
graph_builder.add_node('chat_agent', prompt_llm_chat)
graph_builder.add_node('rag_agent', prompt_llm_rag)
graph_builder.add_node('coding_agent', prompt_llm_code)
graph_builder.add_node('prepare_coding_request', prepare_coding_request)
graph_builder.add_node('accept_coding', accept_coding)


graph_builder.add_edge(START, 'classifier')
graph_builder.add_edge('prepare_coding_request', 'accept_coding')
graph_builder.add_conditional_edges('accept_coding', lambda state: 'end' if state.get('next_node')=='denied' else state['next_node'], {'end':END, 'coding_agent': 'coding_agent', 'accept_coding': 'prepare_coding_request'})
graph_builder.add_conditional_edges('classifier', lambda state: state['message_intent'], {'chat':'chat_agent','knowledge':'rag_agent', 'code':'prepare_coding_request'})
graph_builder.add_edge('chat_agent', END)
graph_builder.add_edge('rag_agent', END)
graph_builder.add_edge('coding_agent', END)

checkpointer=InMemorySaver()
graph =graph_builder.compile(checkpointer)

config= {'configurable':{'thread_id': uuid.uuid4()}}

while True:
    user_message = input('Enter message:')
    result=graph.invoke({'messages':[{'role':'user', 'content': user_message}]}, config=config)

    while '__interrupt__' in result:
        prompt= result['__interrupt__'][0].value
        decision = input(f'{prompt}>')
        result = graph.invoke(Command(resume= decision), config=config)

    print(result['messages'][-1].content)