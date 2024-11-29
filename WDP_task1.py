### Web Data Processing
### Assignment Task 1
### Group 26


import os
import spacy
import replicate
import wikipediaapi

os.environ["REPLICATE_API_TOKEN"] = "r8_NuOLpptj5b1UgnBrexKQkLXdfmd1TZn1TO67O" 

nlp = spacy.load("en_core_web_sm")

# Initialize the Wikipedia API
user_agent = "WDP_group26/v0.1 (b.w.s.kamphorst@student.vu.nl)"
wiki_wiki = wikipediaapi.Wikipedia(user_agent,'en')


# Function to get Wikipedia page URL
def get_wikipedia_url(entity):
    page = wiki_wiki.page(entity)
    if page.exists():
        return page.fullurl
    else:
        return f"No Wikipedia page found for '{entity}'"

# Prompts
pre_prompt = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
prompt_input = input("Enter a prompt: ")


# Generate LLM response
output = replicate.run("replicate/llama-7b:03d3a482ec4f2ec1809171d0ffbd3be7d2a775a01c6bfb5988f4acf39d64f0ce", # LLM model
                        input={"prompt": f"{pre_prompt} {prompt_input} Assistant: ", # Prompts
                        "temperature":0.1, "top_p":0.9, "max_length":128, "repetition_penalty":1})  # Model parameters
     
full_response = ""

for item in output:
    full_response += item


#full_response = "Egypt, in North Africa, is known for its ancient civilization. It's home to the Nile River, the world's longest river, and famous monuments like the Pyramids of Giza, the Sphinx, and the temples of Luxor. Its history dates back over 5,000 years, with the pharaohs, hieroglyphics, and the rise of one of the first complex societies. Cairo, the capital, is a bustling center of culture and history. Egypt's contributions include advances in mathematics, architecture, and medicine. Today, it has a diverse culture blending ancient traditions with modern influences. The economy relies on tourism, agriculture, and the Suez Canal, a key global shipping route."

print("\n \n The LLaMA response is: \n", full_response)

doc = nlp(prompt_input + " " + full_response)

entities = []

for ent in doc.ents:
    entities.append((ent.text,ent.label_,get_wikipedia_url(ent.text)))

entities = list(set(entities))

print("\n \n The recognized entites are: \n")

for entity, category, wikipage in entities:
    print(f"{entity}, {category}, {wikipage}")