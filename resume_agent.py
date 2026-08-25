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
    master_resume: str        # the background, loaded once, the second node after starting
    tailored_resume: str      # LLM-generated markdown/text before PDF render
    output_path: str | None


class FitReasoning(BaseModel):
    is_suitable: bool = Field(..., description= 'Classify whether the master_resume is suitable with the job content'
    'by checking the keyword matches in the job description and the resume and checking graduation dates on both the job description and the resume'
    '.')
    reasoning: str = Field(..., description="2-3 sentences on why, citing specific gaps or matches, also classify the gaps as major or minor")


def load_master_resume(path: str):
    text = open(path, encoding='utf-8').read()
    cleaned = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    if len(cleaned)>0:
        return cleaned

    






graph_builder = StateGraph(ResumeState)