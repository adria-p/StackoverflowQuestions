StackoverflowQuestions
===================

Keyword extraction. More details on
http://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction

How to use:
- Download the training and testing data from the competition and place
it in a 'data' folder (create one if it does not exist).
- Run scripts/deduplicate.py to remove duplicate samples of the training
data set and store the repeated testing set indices.
- Run 'python trainer.py generatePreprocess' to create the tfidf and cv models.
- Run scripts/calculate_distribution.py to create the inverse tag ordering/mapping.
- Run 'python trainer.py' to generate the model.
- Run 'python predictor.py' to generate predictions on the non-repeated test samples.
- Run scripts/zipper.py to generate the final submission file.
