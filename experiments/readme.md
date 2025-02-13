# Description

This folder contains all the scripts necessary to run the experiment presented in the paper. 

The first step is was to create the dataset.
We collected all the necessary datasets and stored the unprocessed datasets in the \data\ folder.
We then ran the data_augmentation.py script which cleaned the datasets, and added language and gibberish detection.
We then the notebook preprocessing_create_datasets.ipnyb to generate the sampled (<1000) datasets for LLMs

Using these datasets we performed all the experiments. 