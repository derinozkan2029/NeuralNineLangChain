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

llm =init_chat_model('anthropic:claude-sonnet-4-6')

class ResumeState(TypedDict):
    job_url: str
    job_description: str    # scraped or cleaned text we will get from LinkedIn, HandShake etc. 
    is_suitable: bool | None # to see if qualifications fit 
    fit_reasoning: str | None
    master_resume: str        # the background, loaded once, the second node after starting
    tailored_resume: str      # LLM-generated markdown/text before PDF render
    output_path: str | None