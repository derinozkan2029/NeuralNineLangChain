import uuid
from typing import TypedDict, Annotated, Literal 
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model # importing this for the LLM model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver #for adding memory to the model

load_dotenv()

llm =init_chat_model('openai: gpt-4.1-mini')

class IntentClassifier(BaseModel):
    message_intent: Literal['chat','knowledge', 'code'] =Field(..., description= 'Classify whether the user' \
    'just wants to chat, ask for knowledge or change code in the project.') #this is for determining which node to go to next

class State(TypedDict): #our own custom state inheriting from TypedDict
    messages = Annotated[list, add_messages]
    message_intent = str | None #this also needs to be passed down from node to node

def classify_intent(state:State): #takes a state and does an LLMP prompt with a structured output
    structured_llm = llm.with_structured_output(IntentClassifier)

    result = structured_llm.invoke([{'role':'system', 'content': 'Determine/classify whether the user wants to chat ("chat) , retrieve a knowledge'
    '("knowledge) or change code ("code)'}, 
    {'role': 'user', 'content': state.messages[-1].content}]) #getting the state and the last message 
    #this creates an instance of an intent classifier 
    return {'message_intent ': result.message_intent}


def prompt_llm_chat(state:State):
    messages = {'role': 'system','content': 'You are a talkative chatbot for fun. Be nice.'} + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]} #bc we are appending we pass this as a list
    # this one is just for chatting

def prompt_llm_rag(state:State):
    messages = {'role': 'system','content': 'No matter what the user says say that "I am a RAG agent."'} + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]} #bc we are appending we pass this as a list

def prompt_llm_code(state:State):
    messages = {'role': 'system','content': 'No matter what the user says say that "I am a CODING agent."'} + state['messages']
    response = llm.invoke(messages)
    return {'messages': [{'role': 'assistant', 'content': response.content}]} #bc we are appending we pass this as a list

