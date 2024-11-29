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



# fill in the file that contains the question in the desired format
fn = "example_input.txt"

# extract questions from the file
question_ids = []
questions = []

# Open the file in read mode
with open(fn, 'r') as file:
    # Read each line in the file
    for line in file:
        # Split the line by the tab character '\t'
        parts = line.strip().split('\t')
        if len(parts) == 2:  # Ensure the line has exactly two parts
            question_id, question = parts
            question = question.removeprefix("Question: ").removesuffix(" Answer:")
            question_ids.append(question_id)
            questions.append(question)




# define the model
llm = Llama(model_path=model_path, verbose=False)

# Generate LLM response based on optimized parameters
for i,id in enumerate(question_id):
    question = questions[i]
    output = llm(
          question, # Prompt
          max_tokens = 32,# Generate up to 32 tokens
          temperature = 0.6,
          top_p = 0.2,
          stop=["Q:"], # Stop generating just before the model would generate a new question
          echo=False #  Do not echo the prompt back in the output
    )

    response = output['choices'][0]['text']

    print("\n \n The LLaMA response is: \n", response)

    doc = nlp(prompt_input + " " + response)

    # extract the entities
    entities = []

    for ent in doc.ents:
        entities.append((ent.text,ent.label_,get_wikipedia_url(ent.text)))

    # keep only unique entities
    entities = list(set(entities))

    # store the results in a file. 
    with open('output.txt', 'w') as output_file:
        output_file.write(f"{id}\tR\"{response}\"\n")
        for entity in entities:
            output_file.write(f"{id}\tE\"{entity}\"\n")
   
