from dotenv import load_dotenv

from langchain.chat_models import init_chat_model # importing this for the LLM model
from langgraph.graph import MessagesState, StateGraph, START, END

load_dotenv()

llm =init_chat_model('anthropic:sonnet-5')
def prompt_llm(state:MessagesState): #the state is passed from one node to the other 
    response =invoke_llm(state['messages']) #this generates a response by invoking the LLM
    return {'messages': [response]} #this doesn't overwrite the messages, it gets added 

graph_builder = StateGraph(MessagesState)
graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, 'prompt_llm') #connecting the nodes we have
graph_builder.add_edge('prompt_llm', END)

graph = graph_builder.compile() #this gives us the graph istance 
user_message = input('Enter message:')
print(graph.invoke({'messages': [{'role' : 'user', 'content': user_message}]}))