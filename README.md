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

It will now take a while before the Llama output to be correctly formatted and written to a txt file called "output.txt". The program wil end after all the output for the output.txt file is created
