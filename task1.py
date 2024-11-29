from llama_cpp import Llama
import os
import sys
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


# Get Wikipedia page URL if it exists
def get_wikipedia_url(entity):
    page = wiki_wiki.page(entity)
    if page.exists():
        return page.fullurl
    else:
        return f"No Wikipedia page found for '{entity}'"



# get the text filename from the command line
fn = sys.argv[1]

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
            question_ids.append(question_id)
            questions.append(question)




# define the model
llm = Llama(model_path=model_path, verbose=False)

# Open output file in append mode to not overwrite each processed question in the file
with open('output.txt', 'a') as output_file:
    # Iterate through all questions
    for i, id in enumerate(question_ids):
        question = questions[i]
        
        
        # Generate LLM response based on optimized parameters
        output = llm(
        question,
        max_tokens=32,
        temperature=0.7,
        top_p=0.2,
        stop=["Q:"],
        echo=False
        )
            
    # Safely extract response
        response = output['choices'][0]['text'].strip()

        # Process entities and extract using spaCy nlp model
        doc = nlp(question + " " + response)
        
        entities = []
        for ent in doc.ents:
            wikipedia_url = get_wikipedia_url(ent.text)
            entities.append((ent.text, ent.label_, wikipedia_url))
        
        # only keep unique entities

        entities = list(set(entities))

        # Write question_id and response to output file
        output_file.write(f"{id}\tR\"{response}\"\n")
        for entity in entities:
            output_file.write(f"{id}\tE\"{entity}\"\n")
        
    
        print(f"Processed {id}")
print("Done processing all questions")
   