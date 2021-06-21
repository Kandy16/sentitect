# sentitect - Sentiment detection using machine learning
* Used LightGBM algorithm
* Used Azure ML python sdk

The development and run environment can be replicated using conda and environment.yml file. 

### Important files to consider

1) 'lightgbm.ipynb'in ds-experiments folder contains the rough model development (Taken as input)
2) 'data' folder contains the sentiment analysis dataset. data-preparation.py creates the CSV files. https://ai.stanford.edu/~amaas/data/sentiment/
3) 'src' folder contains all the codes to be run with azure environment. Uses pipelines.
4) 'run_train.py' amd 'run_predict.py' are the main files to perform training and prediction

Inline comments will be there to assist understanding


The codes in the following links are used to build the code:
https://www.kaggle.com/mehdislim01/simple-yet-efficient-spacy-lightgbm-combination
https://medium.com/@invest_gs/classifying-tweets-with-lightgbm-and-the-universal-sentence-encoder-2a0208de0424

Azure info:
https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-train-models-with-aml
https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-deploy-models-with-aml

https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-batch-scoring-classification
https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-convert-ml-experiment-to-production
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-move-data-in-out-of-pipelines
