from SPARQLWrapper import SPARQLWrapper, JSON
import re
import spacy

nlp = spacy.load("en_core_web_sm")

def parse_question_nlp(question: str):
    """
    Uses NLP to extract entities and relationships from a question.
    """
    doc = nlp(question)
    # Extract named entities and capitalize them
    entities = [ent.text for ent in doc.ents]
    relationships = [token.text for token in doc]

    #for token in doc:
        #if token.dep_ in {"nsubj", "attr", "ROOT", "dobj"}: # Nominal subject, attribute, root, direct object eg. director, of, Pulp Fiction
            #relationships.append(token.text)
        
        
    # what are nsuj, attr, ROOT, dobj?


    # Debug: Print extracted entities and relationships
    print(f"Entities: {entities}")
    print(f"Relationships: {relationships}")
    
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



def fact_check_dynamic(question: str, provided_answer: str):
    """
    Generalized fact-checking function for questions of any shape, using all entities and relationships.
    """
    # Parse the question using NLP
    
    entities, relationships = parse_question_nlp(question)
    if not entities or not relationships:
        return "Unable to parse question."

    # Normalize the provided answer
    normalized_provided = provided_answer.strip().lower()
    print(f"Normalized provided answer: {normalized_provided}")


    # Iterate through all entity-relationship pairs
    for entity in entities:
        for relationship in relationships:
            # Query the knowledge base for this entity-relationship pair
            correct_answers = query_knowledge_base(entity, relationship.lower())
            
            # Skip if no answers found for this combination
            if not correct_answers:
                continue

            # Debug: Print retrieved answers
            
            
            # Normalize and compare answers
            normalized_correct = [answer.split("/")[-1].replace("_", " ").lower() for answer in correct_answers]
            print(f"Normalized correct answers: {normalized_correct}")
            if normalized_provided in normalized_correct:
                return "Correct"
            
            

    # If no match is found after iterating all combinations
    return "Incorrect"

# Example usage
question = " Question: Who is the director of Pulp Fiction? Answer:"
provided_answer = "quentin tarantino"
print(fact_check_dynamic(question, provided_answer))


















def handle_yes_no_question(question, yes_no_answer):
    """
    Handles yes/no questions by verifying the truth of a statement.
    """
    # Parse the question using NLP
    doc = nlp(question)
    entities, relationships = parse_question_nlp(question)
    relationships = [token.text for token in doc]
    if not entities or not relationships:
        return "Unable to parse question."

    expected_value = None
    for token in doc:
        if token.dep_ in {"attr", "dobj", "pobj"}:  # Dependent object, predicate object (direct object), object of preposition  
            expected_value = token.text

    if not expected_value:
        return "Unable to extract expected value."

    for entity in entities:
        for relationship in relationships:
            # Query the knowledge base for this entity-relationship pair
            correct_answers = query_knowledge_base(entity, relationship.lower())
            
            # Skip if no answers found for this combination
            if not correct_answers:
                continue

    # Normalize answers
    normalized_expected = expected_value.strip().lower()
    normalized_correct = [answer.split("/")[-1].replace("_", " ") for answer in correct_answers]
    print(f"Normalized expected value: {normalized_expected}")
    print(f"Normalized correct answers: {normalized_correct}")

    
    
    print(f"Entities: {entities}")
    print (normalized_correct)
    #check if the normalized correct answer is in the list of extracted entities
    for i in normalized_correct:
            if i in entities and yes_no_answer == "Yes":
                return "Correct"
            elif i not in entities and yes_no_answer == "No":
                return "Correct"
            else:
                return "Incorrect"
    # If no match is found after iterating all combinations

# Example usage
question1 = "Is Paris the capital of France?"
print(handle_yes_no_question(question1, "No"))









