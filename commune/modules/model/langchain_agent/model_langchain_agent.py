import commune as c
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType


class WebSurfingAgent(c.Module):
    def __init__(self):
        load_dotenv()
        self.ddg_search = DuckDuckGoSearchResults()
 
        self.HEADERS = {
            'User-Agent': 'Mozila/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90 Gecko/20100101 Firefox/90/0)'
        }
        
        web_fetch_tool = Tool.from_function(
            func=self.fetch_web_page,
            name="WebFetcher",
            description="Fetches the content of a web page"
        )
        
        prompt_template = "Summarize the following content: {content}"
        llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(prompt_template)
        )

        summarize_tool = Tool.from_function(
            func=llm_chain.run,
            name="Summarizer",
            description="Summarizes a web page"
        )
        
        self.tools = [self.ddg_search, web_fetch_tool, summarize_tool]

        self.agent = initialize_agent(
            tools=self.tools,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            llm=llm,
            verbose=True
        )
        
    def fetch_web_page(self, url: str) -> str:
        response = requests.get(url, headers=self.HEADERS)
        return self.parse_html(response.content)

    @staticmethod
    def parse_html(content) -> str:
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text()
    
    def search_the_web(self, query):
        results = self.agent.run(query)
        return results
    
