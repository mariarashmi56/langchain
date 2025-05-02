from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import (create_react_agent, AgentExecutor)
import sys
sys.path.append('/workspaces/langchain/ice_breaker')
from tools.tools import get_profile_url_tavily

def lookup(name :str) -> str:
    """
    This function takes a name as input and returns the LinkedIn profile information of that person.
    """
    llm = ChatOpenAI(temperature=0, model_name = "gpt-4o-mini")

    #Define Prompt
    template = """ Given the full name of the persion {person}, 
    retrieve their LinkedIn profile information profile. 
    The output should only contain their LinkedIn Profile URL."""

    prompt_template = PromptTemplate(input_variables = ["person"], template = template)

    tools_for_agent = [
        Tool(
            name = "Crawl Google 4 LinkedIn Profile URL",
            func = get_profile_url_tavily,
            description = "Useful to search when the LinkedIn profile URL of a person is required."
        )
    ]

    react_prompt = hub.pull("hwchase17/react") 
    agent = create_react_agent(prompt = react_prompt, llm = llm, tools = tools_for_agent)
    agent_executor = AgentExecutor(agent = agent, tools = tools_for_agent, verbose = True)
    result = agent_executor.invoke(input = {"input": prompt_template.format_prompt(person = name)})
    linkedin_profile_url = result["output"]

    return linkedin_profile_url

if __name__ == "__main__":
   
   name = 'Maria Rashmi'
   linkedin_url = lookup(name)
   print(linkedin_url)


