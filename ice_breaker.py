from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import person_intel_parser, PersonIntel

def ice_break(name: str) -> PersonIntel:
    summary_template = """
        given the Linkedin information {information} about a person I want you to create
        1. a short summary of the person
        2. two interesting facts about them
        3. a topic that may interest them
        4. 2 creative ice breakers to open a conversation with them
        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], 
        template=summary_template,
        partial_variables={"format_instructions": person_intel_parser.get_format_instructions()})    

    llm = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url)
    linkedin_slimmed = {
        "name": linkedin_data["full_name"],
        "occupation": linkedin_data["occupation"],
        "headline": linkedin_data["headline"],
        "summary": linkedin_data["summary"],
        "location": linkedin_data["country_full_name"],
        "city": linkedin_data["city"],
        "state": linkedin_data["state"],
        "experience": linkedin_data["experiences"],
        "education": linkedin_data["education"]
    }

    result = chain.run(information=linkedin_slimmed)
    print(result)
    return person_intel_parser.parse(result)

if __name__ == "__main__":
    result = ice_break(name="Eden Marco")
