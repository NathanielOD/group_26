from llama_cpp import Llama
import os
import spacy
import wikipediaapi

model_path = "models/llama-2-7b.Q4_K_M.gguf"

### Assignment Task 1
### Group 26


#load spacy model for entity recognition
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
prompt_input = ("What is the capital of Egypt?")




llm = Llama(model_path=model_path, verbose=False)

# Generate LLM response based on optimized parameters
output = llm(
      prompt_input, # Prompt
      max_tokens = 32,# Generate up to 32 tokens
      temperature = 0.6,
      top_p = 0.2,
      stop=["Q:"], # Stop generating just before the model would generate a new question
      echo=False #  Do not echo the prompt back in the output
)

full_response = ""



print("\n \n The LLaMA response is: \n", output['choices'][0]['text'])

doc = nlp(prompt_input + " " + output['choices'][0]['text'])

entities = []

for ent in doc.ents:
    entities.append((ent.text,ent.label_,get_wikipedia_url(ent.text)))

entities = list(set(entities))

print("\n \n The recognized entites are: \n")

for entity, category, wikipage in entities:
    print(f"{entity}, {category}, {wikipage}")
