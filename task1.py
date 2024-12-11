### Web Data Processing
### Assignment Task 1
### Group 26

# Import all necessary packages
from llama_cpp import Llama
import os
import sys
import spacy
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests

# Initialise the LLM
model_path = "models/llama-2-7b.Q4_K_M.gguf"
'''
model_path = "llama-2-7b.Q4_K_M.gguf"
if not os.path.exists(model_path):
    !wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf -O {model_path} # not necessary in Docker code
'''
llm = Llama(model_path=model_path, verbose=False)


# Load the spacy model
nlp = spacy.load("en_core_web_sm")

# and the wordnet database and punctuality and stopwords datasets
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Function to get the context of an entity mention
def get_ent_context(entity, text, window_size=5):
    # Tokenize response and remove punctuation
    tokens = word_tokenize(re.sub(r'[^\w\s]', '', text))
    tokens = [token.lower() for token in tokens]

    # Check if the entity consists of multiple words
    entity_words = entity.lower().split()

    # Find the index of the first word of the entity
    try:
        entity_index = tokens.index(entity_words[0])
    except ValueError:
        return []  # Entity not found in tokens

    # Find window of text around the entity
    start = max(entity_index - window_size, 0)
    end = min(entity_index + len(entity_words) + window_size, len(tokens))

    # Extract context without stop words
    stop_words = set(stopwords.words('english'))
    context = [token for i, token in enumerate(tokens[start:end])
               if i < len(entity_words) and token not in stop_words]

    return context

# Search Wikipedia for an entity, return all pages. These are the candidates for entity linking
def search_wikipedia_pages(entity):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": entity,
        "format": "json",
    }

    response = requests.get(url, params=params)
    data = response.json()
    search_results = data["query"]["search"]

    # Extract titles and page IDs
    pages = [
        {
            "title": result["title"],
            "page_id": result["pageid"]
        }
        for result in search_results
    ]

    return pages

# Function to obtain the summary of a Wikipedia page
def get_wikipedia_summary(page_id):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "pageids": page_id,  # Use the page ID
        "prop": "extracts",
        "exintro": True,  # Only get the introduction (summary)
        "explaintext": True,  # Get plain text, not HTML
        "format": "json",
    }

    response = requests.get(url, params=params)
    data = response.json()
    page = data["query"]["pages"].get(str(page_id))

    if page and "extract" in page:
        return {
            "title": page["title"],
            "summary": page["extract"]
        }
    else:
        return None

# Function to compute the Jaccard Similarity
def compute_JS(tokens1,tokens2):
    # Convert lists to sets
    set1 = set(tokens1)
    set2 = set(tokens2)

    # Compute intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate Jaccard Similarity
    return len(intersection) / len(union) if union else 0

# Function to remove incorrect/incomplete entities
def filter_incomplete_entities(entities):
    filtered_entities = []

    for entity in entities:
        entity_name = entity[0]
        if len(entity_name) > 3 and entity_name.isalnum():
            filtered_entities.append(entity)

    return filtered_entities

# get the text filename from the command line
text_file = sys.argv[1]

# extract questions from the file
question_ids = []
questions = []


# Open the file in read mode
with open(text_file, 'r') as file:
    # Read each line in the file
    for line in file:
        # Split the line by the tab character '\t'
        parts = line.strip().split('\t')
        if len(parts) == 2:  # Ensure the line has exactly two parts
            question_id, question = parts
            question_ids.append(question_id)
            questions.append(question)

# Compute the ouput and write it in a text file
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

        # Extract response from LLM
        response = output['choices'][0]['text'].strip()

        # Extract the entities using Spacy NLP model
        doc = nlp(question + " " + response)

        # Find the Wikipedia page for each entity and store them together
        entities = []
        for ent in doc.ents:
            # Obtain context of entity mention
            context = get_ent_context(ent.text, doc.text)

            # Search for the entity in wikipedia, obtain all candidates
            pages = search_wikipedia_pages(ent.text)

            if not pages:
                wikipedia_url = None
            else:
                JS_score = []

                # Go over each candidate and compute its similarity with the entity mention
                for p in pages:
                    title = p["title"]
                    page_id = p["page_id"]

                    # Obtain summary of Wikipedia page and tokenize it, remove stop words and interpunction
                    summary = get_wikipedia_summary(page_id)['summary']
                    summ_tokens = word_tokenize(re.sub(r'[^\w\s]', '', summary))
                    summ_tokens = [token.lower() for token in summ_tokens]
                    stop_words = set(stopwords.words('english'))
                    summ_tokens = [token for token in summ_tokens if token not in stop_words]
                    JS_score.append(compute_JS(context,summ_tokens))

                # Determine the best match as the one with the highest JS score
                best_page = JS_score.index(max(JS_score))
                page_id = pages[best_page]["page_id"]

                wikipedia_url = f"https://en.wikipedia.org/?curid={page_id}"


            entities.append((ent.text, ent.label_, wikipedia_url))

        # Filter incorrect/incomplete entities
        entities = filter_incomplete_entities(entities)

        # Get rid of duplicate entities
        entities = list(set(entities))

        # Write question_id and response to output file
        output_file.write(f"{id}\tR\"{response}\"\n")
        for entity in entities:
            output_file.write(f"{id}\tE\"{entity[0]}\"\t\"{entity[2]}\"\n")
   
