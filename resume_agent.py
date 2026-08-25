import uuid
import os
import subprocess
import requests
import re
from typing import TypedDict, Annotated, Literal 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup


from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  #this is for intelligent embeddings
from langchain.chat_models import init_chat_model # importing this for the LLM model
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver #for adding memory to the model
from langgraph.types import interrupt,Command 

load_dotenv()

llm =init_chat_model('anthropic:claude-sonnet-4-6')

class ResumeState(TypedDict):
    job_url: str
    job_description: str    # scraped or cleaned text we will get from LinkedIn, HandShake etc. 
    is_suitable: bool | None # to see if qualifications fit 
    fit_reasoning: str | None
    master_resume: str   # the background, loaded once, the second node after starting
    tailored_resume: str # LLM generated text before PDF render
    output_path: str | None


class FitReasoning(BaseModel):
    is_suitable: bool =Field(..., description= 'Classify whether the master_resume is suitable with the job content'
    'by comparing the required qualifications and actual experience () and checking graduation dates on both the job description and the resume'
    '.')
    reasoning: str = Field(..., description="2-3 sentences on why, citing specific gaps or matches, also classify the gaps as major or minor")

def assess_fit(state:ResumeState): #takes a state and does an LLMP prompt with a structured output
    structured_llm = llm.with_structured_output(FitReasoning)

    result = structured_llm.invoke([{'role':'system', 'content': 'Determine/classify whether the job description is suitable with the master resume.'}, 
    {'role': 'user', 'content': state['master_resume'] state['job_description']}]) #getting the job description and the master resume
    #this creates an instance of a fit reasoning 
    return {'is_suitable': result.is_suitable, 'fit_reasoning': result.fit_reasoning}


def load_master_resume(path: str):
    text = open(path, encoding='utf-8').read()
    cleaned = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    if len(cleaned)>0:
        return cleaned

def fetch_job_posting(state:ResumeState):
    posting = requests.get(state['job_url'], headers={'User-Agent': '...'})
    posting.raise_for_status()
    soup=BeautifulSoup(posting.text, 'html.parser')
    description_div = soup.find('div', class_='show-more-less-html__markup')
    return {'job_description': description_div.get_text(strip=True) if description_div else ''} #returning a dictionary

def curate_tailored_resume(state:ResumeState):


    

graph_builder = StateGraph(ResumeState)