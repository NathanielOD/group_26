# group_26

 README

Install the packages that are included in the requirements.txt file.
pip install -r ./workspace/requirements.txt

Afterwards, you need to run an additional line to downlaod the spacy nlp model which is used for entitiy recognition. Run the lines below:
python -m spacy download en_core_web_sm
Your text file should be in the same directory as the task1 python script.

You can now run the code in the Docker container by using the commands:
python ./workspace/task1.py[YOUR TEXT FILE]
or
python /workspace/task1.py[PATH TO YOUR TEXT FILE]

It will now take a while before the Llama output to be correctly formatted and written to a txt file called "output.txt". After a question is processed, the system will notify you. After all the processing is done the program will end and notify you. 
