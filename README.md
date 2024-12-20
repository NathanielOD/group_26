# group_26

 README

Install the packages that are included in the requirements.txt file.
pip install -r ./workspace/requirements.txt

Afterwards, you need to run an additional line to downlaod the spacy nlp model which is used for entitiy recognition. Run the lines below:
python -m spacy download en_core_web_sm
Your text file should be in the same directory as the final_task python script.

You can now run the code in the Docker container by using the commands:
python ./workspace/final_task.py ./workspace/[YOUR TEXT FILE]
or
python /workspace/final_task.py[PATH TO YOUR TEXT FILE]

If the program encounters network issue please run the following python ./workspace/final_task_without_SPARQL.py ./workspace/[YOUR TEXT FILE]
This version of the program does not make use of the fact checking function which queries dbpedia.

It will now take a while before the Llama output to be correctly formatted and written to a txt file called "output.txt". The program wil end after all the output for the output.txt file is created

When running in the docker container on docker desktop, make sure latest version of docker desktop is installe. If not, SPARQL queries may not work due to a network compatibility in docker
