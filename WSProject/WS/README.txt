This README describes the format of the data in the web science project dataset. 
The data is from stack exchange skeptics, a question-answering site for scientific skepticism. 

########### CROWDSOURCING AND CLASSIFIERS

`web_science_dataset.jsonl`
The file "web_science_dataset.jsonl" contains one json
object per line, which has the following fields:

questionId: The ID of the question on stack exchange skeptics
question: The question text
questionUrl: A link to the original question on stack exchange skeptics
category: The category of the question in text form
categoryId: An Id for the category
answerId: The ID of the answer on stack exchange skeptics
answer: The answer text
answerUrl: A link to the original answer on stack exchange skeptics

`web_science_dataset_with_labels.jsonl` 
Same format as above, but with two added fields: "answerLabel", "answerQuality"

`crowdsourced_data.zip`
This is a zip file containing the csv's from our crowdsourcing assignment. All results have been anonymized. You will use this for the assignment detailed in Section 2.2 of the "WebScienceFinalProjectDescription.pdf" document.


`training_ids.txt` and `testing_ids.txt`
These two files contain the list of the "questionId"s for training and testing, respectively.  


########### RECOMMENDER SYSTEM 

`recommender_train.tsv` and `recommender_test.tsv`
These two tsv's are what you will use to train your recommender system.
The training file has 3 columns: ["userID","questionID","rating"]
The test file has 3 columns: ["userID","questionID","recommend"]
