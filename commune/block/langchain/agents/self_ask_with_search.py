from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
import commune

import os
os.environ['OPENAI_API_KEY'] = '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'
os.environ['SERPAPI_API_KEY'] = '19a8a494729d0a843e9c162e1c8a3e2de5c6c943aa6e3dcdea2d8e7e9a063dc7'
def AgentModule():
    llm = OpenAI(temperature=0, model_name='text-babbage-001')
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Intermediate Answer",
            func=search.run
        )
    ]

    self_ask_with_search = initialize_agent(tools, llm, agent="self-ask-with-search", verbose=True)
    
    return commune.module(self_ask_with_search, serve=False)
    

if __name__ == "__main__":
    
    AgentModule().serve(name='agent_slamy')