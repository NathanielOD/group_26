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
               if token not in stop_words]

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
# takes as input a page id or a URL
def get_wikipedia_summary(input_value):
    url = "https://en.wikipedia.org/w/api.php"

    # Check if input is a URL
    if isinstance(input_value, str) and input_value.startswith("http"):
        # Extract the page ID from the URL
        match = re.search(r"curid=(\d+)", input_value)
        if match:
            page_id = match.group(1)
        else:
            raise ValueError("The URL does not contain a valid page ID.")
    else:
        # Assume input is the page ID
        page_id = str(input_value)

    # Request parameters
    params = {
        "action": "query",
        "pageids": page_id,  # Use the page ID
        "prop": "extracts",
        "exintro": True,  # Only get the introduction (summary)
        "explaintext": True,  # Get plain text, not HTML
        "format": "json",
    }

    # Send the request
    response = requests.get(url, params=params)
    data = response.json()

    # Access the page content
    page = data["query"]["pages"].get(page_id)

    if page and "extract" in page:
        return {
            "title": page["title"],
            "summary": page["extract"]
        }
    else:
        return None

def is_disambiguation_page(page_id):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "categories",
        "pageids": page_id
    }
    response = requests.get(url, params=params).json()
    categories = response.get("query", {}).get("pages", {}).get(str(page_id), {}).get("categories", [])
    for category in categories:
        if "Disambiguation pages" in category.get("title", ""):
            return True
    return False

def get_first_result_page(disambig_page_id):
    # Step 1: Get the links on the disambiguation page
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "prop": "links",
        "pageids": disambig_page_id,
        "pllimit": 1  # Limit to the first link
    }
    response = requests.get(url, params=params).json()
    links = response.get("query", {}).get("pages", {}).get(str(disambig_page_id), {}).get("links", [])
    if not links:
        return None  # No links found

    first_title = links[0]["title"]

    # Step 2: Get the page ID of the first link
    params = {
        "action": "query",
        "format": "json",
        "titles": first_title
    }
    response = requests.get(url, params=params).json()
    pages = response.get("query", {}).get("pages", {})
    first_page_id = next(iter(pages.values())).get("pageid", None)

    return f"https://en.wikipedia.org/?curid={first_page_id}"

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
        entity_name_wo_whitespace = entity_name.replace(" ","")
        if len(entity_name) > 3 and entity_name_wo_whitespace.isalnum():
            filtered_entities.append(entity)

    return filtered_entities

# Performs Named Entity Recognition on a corpus of text
# Returns array with entities and corresponding wikipedia page
def NER(text,top_wiki_page=True):
  doc = nlp(text)
  entities = []

  for ent in doc.ents:
     # Obtain context of entity mention
    context = get_ent_context(ent.text, doc.text)

    # Search for the entity in wikipedia, obtain all candidates
    pages = search_wikipedia_pages(ent.text)
    pages = pages[:min(3,len(pages))] # take only the three top pages

    if not pages:
        wikipedia_url = None
    else:
        if(top_wiki_page):
          page_id = pages[0]["page_id"]
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

        if is_disambiguation_page(page_id):
          wikipedia_url = get_first_result_page(page_id)
        else:
          wikipedia_url = f"https://en.wikipedia.org/?curid={page_id}"

    cand = [ent.text, wikipedia_url]
    if cand not in entities:
      entities.append(cand)
  return entities


def is_yesno_question(response, entities_response, entities_question):
  result = [False]

  for ent in entities_response:
    if ent not in entities_question:
      return result

  # Tokenize and remove stop words and interpunction
  response_tokens = word_tokenize(re.sub(r'[^\w\s]', '', response))
  response_tokens = [token.lower() for token in response_tokens]
  response_tokens = [token for token in response_tokens if token not in stop_words]

  # Compute JS of the response with affirmative and negative words
  JS_affirmative = compute_JS(affirmative,response_tokens)
  JS_negative = compute_JS(negative,response_tokens)

  # Determine whether the question is yes/no
  if max(JS_affirmative,JS_negative) >= threshold:
      result[0] = True
      if JS_affirmative > JS_negative:
          result.append("yes")
      else:
          result.append("no")

  return result

def determine_entity_answer(entities_question,entities_response):

  if len(entities_response)==0:
    return None

  result = entities_response[0][0]#[1]
  if len(entities_response) == 1:
    return result
  else:
    # construct a string of all wiki summaries of the entities in the question
    summaries = ""
    for ent in entities_question:
      summary = get_wikipedia_summary(ent[1])
      if summary and 'summary' in summary:
          summaries += summary['summary']

    # convert to tokens without interpunction and stop words
    summaries_tokens = word_tokenize(re.sub(r'[^\w\s]', '', summaries))
    summaries_tokens = [token.lower() for token in summaries_tokens if token not in stop_words]

    JS_max = 0

    # create list of possible answers (all entities in the response)
    pos_answers = [ent[0] for ent in entities_response]
    pos_answers = list(set(pos_answers))


    # now, we check for each entity in the response the similarity with the merged wiki summaries of the entities in the question
    # the entity in the response with the highest JS with the merged summaries wins
    for ans in pos_answers:
      JS_current = compute_JS(word_tokenize(ans.lower()), summaries_tokens)

      if JS_current >= JS_max:
        result = ans
        JS_max = JS_current

  return result


# Function to fact check yes/no answer using wikipedia
def check_binary_answer(question, answer):
    """
    Verify a yes/no answer using Wikipedia content.

    Parameters:
    - question: The original question.
    - answer: The yes/no answer generated.

    Returns:
    - True if the answer aligns with Wikipedia content, False otherwise.
    """
    search_results = search_wikipedia_pages(question)
    correctness = False
    best_jaccard_score = 0

    question_tokens = set(word_tokenize(question.lower()))

    for result in search_results:
        page_content = get_wikipedia_summary(result["page_id"])
        if not page_content:
            continue

        # Tokenize page content
        doc = nlp(page_content['summary'])
        sentences = [set(word_tokenize(sent.text.lower())) for sent in doc.sents]

        # Validate yes/no answer based on similarity with Wikipedia content
        for sentence_tokens in sentences:
            

            jaccard_score = compute_JS(question_tokens, sentence_tokens)
            if jaccard_score > best_jaccard_score:
                best_jaccard_score = jaccard_score
            
            

            if answer.lower() == "yes" and best_jaccard_score > 0.2:
                correctness = True
                
            elif answer.lower() == "no" and best_jaccard_score < 0.08:
                
                correctness = True

    return correctness


# Function to fact check open answer using wikipedia
def check_entity_answer(question, answer):
    """
    Similar to check_binary_answer, only for entities
    """
    search_results = search_wikipedia_pages(question)
    correctness = False
    best_jaccard_score = 0
    answer_tokens = set(word_tokenize(answer.lower()))

    for result in search_results:
        page_content = get_wikipedia_summary(result["page_id"])
        if not page_content:
            continue

        doc = nlp(page_content['summary'])
        sentences = [set(word_tokenize(sent.text.lower())) for sent in doc.sents]

        # Validate open answer based on similarity with Wikipedia content
        for sentence_tokens in sentences:
            

            jaccard_score = compute_JS(answer_tokens, sentence_tokens)
            if jaccard_score > best_jaccard_score:
                best_jaccard_score = jaccard_score
            if best_jaccard_score > 0.25:
                
                correctness = True

    return correctness


def parse_question_nlp(question: str):
    
    doc = nlp(question)

    entities = [ent.text for ent in doc.ents]
    relationships = [token.text for token in doc]




    
    return entities, relationships


def query_knowledge_base(entity: str, relationship: str):
    """
    Queries DBpedia to retrieve the correct answer for the given entity and relationship.
    """
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    # Create the SPARQL query based on possible relationship and entity
    sparql.setQuery(f"""
    SELECT ?answer WHERE {{
      dbr:{entity.replace(' ', '_')} dbo:{relationship} ?answer .
    }}
    """)
    sparql.setReturnFormat(JSON)
    
    results = sparql.query().convert()
    
    return [result["answer"]["value"] for result in results["results"]["bindings"]]



def fact_check_entity(question, provided_answer):
    
    # Parse the question using NLP
    
    entities, relationships = parse_question_nlp(question)
    if not entities or not relationships:
        return "Unable to parse question."

    # Normalize the provided answer
    normalized_provided = provided_answer.strip().lower()
    


    # Iterate through all entity-relationship pairs
    for entity in entities:
        for relationship in relationships:
            # Query the knowledge base for this entity-relationship pair
            correct_answers = query_knowledge_base(entity, relationship.lower())
            
            # Skip if no answers found for this combination
            if not correct_answers:
                continue
            
            # Normalize and compare answers
            normalized_correct = [answer.split("/")[-1].replace("_", " ").lower() for answer in correct_answers]


            if normalized_provided in normalized_correct:
                return True
            
            #if no match is found after iterating all combinations return no match


    
    return False





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


# CONSTANTS

affirmative = ["yes", "certainly", "definitely", "indeed", "obviously", "clearly",
              "positively", "agreed", "absolutely", "undoubtedly", "surely",
              "in fact", "right", "true", "affirmative", "always","everybody knows"]

negative = ["no", "not", "never", "unlikely", "doubt", "doubtful", "impossible",
            "cannot", "wrong", "refuse", "denied", "negative", "never", "disagree",
            "contrary", "incorrect", "mistaken", "out of the question", "reject",
            "false", "absent"]

threshold = 0.03

stop_words = set(stopwords.words('english'))

# Compute the ouput and write it in a text file
# Compute the ouput and write it in a text file
with open('output.txt', 'a') as output_file:
    # Iterate through all questions
    for i, id in enumerate(question_ids):
        # Check for duplicates
        '''
        if id in processed_questions:
              continue
        processed_questions.add(id)
        '''

        question = questions[i]

        # Generate LLM response based on optimized parameters
        output = llm(
        question,
        max_tokens=25,
        temperature=0.2,#testing different temperatures to see how the answer extracting responds
        top_p=0.2,
        stop=["Q:", "Question:"], # geen \n want zo begint hij altijd zn antwoord over nicaragua
        #top_k=10,
        echo=False
        )

        # Extract response from LLM
        response = output['choices'][0]['text'].strip()

        #### ENTITY RECOGNITION AND LINKING ####

        # Find the Wikipedia page for each entity and store them together
        entities_question = NER(question)
        entities_response = NER(response)

        # Merge the entities in question and response and get rid of duplicates
        entities = entities_question + entities_response
        entities = [list(x) for x in set(tuple(lst) for lst in entities)]


        #### EXTRACTING ANSWER (YES/NO OR ENTITY) ####

        yesno = is_yesno_question(response, entities_response, entities_question)
        if yesno[0]:
          answer = yesno[1]
        else:
          answer = determine_entity_answer(entities_question,entities_response)

        if answer == None: # Try once more
          # Extract response from LLM
          response = output['choices'][0]['text'].strip()
          # Find the Wikipedia page for each entity
          entities_response = NER(response)
          # Merge the entities in question and response and get rid of duplicates
          entities = entities_question + entities_response
          entities = [list(x) for x in set(tuple(lst) for lst in entities)]
          # Extract answer
          yesno = is_yesno_question(response, entities_response, entities_question)
          if yesno[0]:
            answer = yesno[1]
          else:
            answer = determine_entity_answer(entities_question,entities_response)

        #### FACT CHECKING ####

        if yesno[0]:  # Yes/no answer
          if check_binary_answer(question, answer):
            correctness = "correct"
          else:
            correctness = "incorrect"
        else:  # Open answer
          if fact_check_entity(question, answer) is True:
            correctness = "correct"
          elif fact_check_entity(question, answer) is False:
            correctness = "incorrect"
          elif fact_check_entity(question, answer) == "Unable to parse question.":
            if check_entity_answer(question, answer):
              correctness = "correct"
            else:
              correctness = "incorrect"

        # Write question_id and response to output file
        output_file.write(f"{id}\tR\"{response}\"\n")
        output_file.write(f"{id}\tA\"{answer}\"\n")
        output_file.write(f"{id}\tC\"{correctness}\"\n")
        for entity in entities:
            output_file.write(f"{id}\tE\"{entity[0]}\"\t\"{entity[1]}\"\n")
   
