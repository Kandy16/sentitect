# sentitect - Sentiment detection using machine learning
* Used LightGBM algorithm
* Used Azure ML python sdk

The development and run environment can be replicated using conda and environment.yml file. 

### Important files to consider

1) 'lightgbm.ipynb'in ds-experiments folder contains the rough model development (Taken as input)
2) 'data' folder contains the sentiment analysis dataset. data-preparation.py creates the CSV files. https://ai.stanford.edu/~amaas/data/sentiment/
3) 'src' folder contains all the codes to be run with azure environment. Uses pipelines.
4) 'run_train.py' amd 'run_deploy.py' are the main files to perform training and deployment

Inline comments will be there to assist understanding
