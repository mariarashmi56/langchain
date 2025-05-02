import os
import requests
from dotenv import load_dotenv

load_dotenv()


#sujith "https://gist.githubusercontent.com/mariarashmi56/67c5993d62af4a21b9b6c282c48c2110/raw/d5799d66ce86fb2718764d86f6a4648a47560aef/gistfile1.txt"
#maria "https://gist.githubusercontent.com/mariarashmi56/ce6d9369e2c9a1e822593e476a245d02/raw/4031d53b71fd0f5b4dd94f323104d1858a23d1e3/gistfile1.txt"

def scrape_linkedin_profile(linkedin_profile_url : str, mock : bool = True) -> dict:
    """ Scrape LinkedIn profile information from a given URL."""
    if mock:
        linkedin_profile_url = "https://gist.githubusercontent.com/mariarashmi56/ce6d9369e2c9a1e822593e476a245d02/raw/4031d53b71fd0f5b4dd94f323104d1858a23d1e3/gistfile1.txt"
        response = requests.get(linkedin_profile_url, timeout= 15)
    else:
        api_endpoint = "https://api.scrapin.io/enrichment/profile"
        params = {
            "apikey": os.environ["SCRAPIN_API_KEY"],
            "linkedInUrl": linkedin_profile_url,
        }

        response = requests.get(api_endpoint, params=params, timeout= 15)


    
    data = response.json().get("person")
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None) and k not in ["certifications"]
    }

    return data

