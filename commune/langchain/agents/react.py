from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents.react.base import DocstoreExplorer
import commune

import os
os.environ['OPENAI_API_KEY'] = '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'
def ReactAgentModule():

    docstore=DocstoreExplorer(Wikipedia())
    tools = [
        Tool(
            name="Search",
            func=docstore.search
        ),
        Tool(
            name="Lookup",
            func=docstore.lookup
        )
    ]

    llm = OpenAI(temperature=0)
    react = initialize_agent(tools, llm, agent="react-docstore", verbose=True)

    
    return commune.module(react)
    

if __name__ == "__main__":
    
    module = ReactAgentModule()
    # question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
    # module.run(question)
    # # module = AgentModule()
    
    module = module.serve(name='ReactAgentModule')
    # print(result)


# llm = OpenAI(temperature=0, model_name="text-davinci-002")
# react = initialize_agent(tools, llm, agent="react-docstore", verbose=True)