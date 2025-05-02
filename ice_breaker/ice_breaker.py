from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from third_parties.linkedIn import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup
from output_parsers import summary_parser, Summary

def ice_breaker_with(name: str) -> tuple[Summary, str]:
    """
    This function takes a name as input and returns the LinkedIn profile information of that person.
    """
    # Get the LinkedIn profile URL
    linkedin_profile_url = lookup(name)
    
    # Scrape the LinkedIn profile information
    linkedIn_profile = scrape_linkedin_profile(linkedin_profile_url, mock=True)

    summary_template = """ Given the linkedIn information {information}, 
        1. generate a short summary 
        2. tell me two interesteting facts about this person 
        \n{format_instructions}
        """
    
    prompt = PromptTemplate(input = ['information'], 
                            template=summary_template, 
                            partial_variables = {'format_instructions': summary_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)

    chain = prompt | llm | summary_parser

    res:Summary = chain.invoke({"information": linkedIn_profile})
    
    return res, linkedIn_profile.get('profile_pic_url')


if __name__ == '__main__':
    print("Ice Breaker Enter")
    name = "Maria Rashmi"
    ice_breaker_with(name)
 
    
    


    
    
    
    
    

    
  



