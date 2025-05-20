from dotenv import load_dotenv
from langchain.agents import tool
from typing import Union
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.agents.agent import AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def get_text_len(text:str) -> int:
    """
    This function takes a string as input and returns the length of the string.
    """
    text = text.strip("'\n").strip('"')
    return len(text)

def find_tool_name(tools, tool_name):
    """
    This function takes a list of tools and a tool name as input and returns the tool object.
    """
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool {tool_name} not found in tools list")


if __name__ == "__main__":
    print("Hello React Agent")
    tools = [get_text_len]

    react_prompt = """ Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
        """
    prompt = PromptTemplate(template = react_prompt).partial(
        tools = render_text_description(tools) , 
        tool_names = ",".join([tool.name for tool in tools])
    )
    #initialize LLM

    llm = ChatOpenAI(
        temperature = 0, 
        model = "gpt-3.5-turbo", 
        stop=["\nObservation", "Observation"])
    intermediate_steps = []

    agent = {
        "input": lambda x :x["input"], 
        "agent_scratchpad" : lambda x : format_log_to_str(x['agent_scratchpad'])
        } | prompt | llm | ReActSingleInputOutputParser()

    #invoke the agent
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {"input" : "What is the length in characters of the string 'dog' ?", 
         "agent_scratchpad" : intermediate_steps
         }
        )

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tools_to_use = find_tool_name(tools, tool_name)
        tool_input = agent_step.tool_input
        observation = tools_to_use.func(str(tool_input))
        print(f"Observation: {observation}")
        intermediate_steps.append((agent_step, str(observation)))

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {"input" : "What is the length in characters of the string 'dog'?", 
         "agent_scratchpad" : intermediate_steps}
        )
    if isinstance(agent_step, AgentFinish):
        print("Final Answer: ", agent_step.return_values)



                     