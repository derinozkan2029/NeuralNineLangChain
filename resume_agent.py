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
    fit_reasoning: str = Field(..., description="2-3 sentences on why, citing specific gaps or matches, also classify the gaps as major or minor")

def assess_fit(state:ResumeState): #takes a state and does an LLMP prompt with a structured output
    structured_llm = llm.with_structured_output(FitReasoning)

    result = structured_llm.invoke([{'role':'system', 'content': 'Determine/classify whether the job description is suitable with the master resume.'}, 
    {'role': 'user', 'content': f"MASTER RESUME:\n{state['master_resume']}\n\nJOB DESCRIPTION:\n{state['job_description']}"}]) #getting the job description and the master resume
    #this creates an instance of a fit reasoning 
    return {'is_suitable': result.is_suitable, 'fit_reasoning': result.fit_reasoning}


def load_master_resume(path: str):
    text = open(path, encoding='utf-8').read()
    cleaned = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    if len(cleaned)>0:
        return cleaned

def fetch_job_posting(state:ResumeState):
    posting = requests.get(state['job_url'], headers={'User-Agent': '...'})
    posting.raise_for_status() # check the HTML 200
    soup=BeautifulSoup(posting.text, 'html.parser') 
    description_div = soup.find('div', class_='show-more-less-html__markup')
    return {'job_description': description_div.get_text(strip=True) if description_div else ''} #returning a dictionary

def curate_tailored_resume(state: ResumeState):
    messages = [ {'role': 'system', 'content': (
            'Only use experience, skills, and accomplishments present in the master resume; '
            'do not invent qualifications, metrics, or experience not in the source material. '
            'Reorder bullets, reweight which experiences are emphasized, and adjust the professional '
            'summary to fit the job. Mirror the job posting\'s terminology where honestly applicable '
            '(e.g. if the posting says "distributed systems" and the resume has genuinely relevant work, '
            'use that phrasing). Follow the same markdown section headers as the master resume, in the '
            'same order, mirrored exactly. Keep total content to what fits one page.')},
        {'role': 'user', 'content': (  f"MASTER RESUME:\n{state['master_resume']}\n\n" f"JOB DESCRIPTION:\n{state['job_description']}"
        )}, ]
    response = llm.invoke(messages)
    return {'tailored_resume': response.content}

def report_not_suitable(state: ResumeState):
    return {'output_path': None}

def render_pdf(state: ResumeState):
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace', 'templates', 'resume.css') 
    css = open(css_path, encoding='utf-8').read()
    # this builds the path to the CSS file copied earlier, open it, read its raw text into a string. 

    full_html = f"""<html>
<head>
<meta charset="utf-8">
<style>
{css}
</style>
</head>
<body>
{state['tailored_resume']}
</body>
</html>"""

    # an f-string building one complete HTML document as text here.

    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace', 'tmp', 'tailored.html')
    # where we are going to save that HTML string as an actual file, so Chrome can open it.
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    #creates the workspace/tmp/ folder if it doesn't exist yet
    with open(html_path, 'w', encoding='utf-8') as f: #writes the HTML string to disk at html_path
        f.write(full_html)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace', 'output', 'tailored_resume.pdf')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    subprocess.run([
        chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={output_path}",
        f"file://{html_path}"
    ], check=True)

    #launches Chrome as a separate program  in headless mode , tells it to load the HTML file we just wrote (file://{html_path}) and print it straight to a PDF at output_path, then exits.

    return {'output_path': output_path}



graph_builder = StateGraph(ResumeState)
graph_builder.add_node('assess_fit', assess_fit)
graph_builder.add_node('fetch_job_posting', fetch_job_posting)
graph_builder.add_node('tailor_resume', curate_tailored_resume)
graph_builder.add_node('render_pdf', render_pdf)


graph_builder.add_edge(START, 'fetch_job_posting')
graph_builder.add_edge('fetch_job_posting', 'assess_fit')
graph_builder.add_conditional_edges('assess_fit', lambda state: 'suitable' if state['is_suitable'] else 'not_suitable', {'suitable': 'tailor_resume', 'not_suitable': 'report_not_suitable'})
# if not suitable we don't need to run it or tailor the resume
graph_builder.add_edge('report_not_suitable', END)
graph_builder.add_edge('tailor_resume', 'render_pdf')
graph_builder.add_edge('render_pdf', END)

graph = graph_builder.compile()

resume_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace', 'resume_master.md')
master_resume = load_master_resume(resume_path)

job_url = input('Enter job posting URL: ')

initial_state = {
    'job_url': job_url, 'job_description': '', 'is_suitable': None, 'fit_reasoning': None,
    'master_resume': master_resume, 'tailored_resume': '', 'output_path': None,
}
result = graph.invoke(initial_state) #no thread_id this time bc I didn't have a checkpointer
if result.get('output_path'):
    print(f"Resume written to {result['output_path']}")
else:
    print(f"Not a good fit: {result['fit_reasoning']}")