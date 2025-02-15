# Description

This folder contains all the scripts necessary to run the experiment presented in the paper. 

The first step is was to create the dataset.
We collected all the necessary datasets and stored the unprocessed datasets in the \data\ folder.
We then ran the data_augmentation.py script which cleaned the datasets, and added language and gibberish detection.
We then the notebook preprocessing_create_datasets.ipnyb to generate the sampled (<1000) datasets for LLMs

Using these datasets we performed all the experiments. 

# Experiments: 

There are 3 components to the experiment: 
- experiment.py (example of parameters in distil.sh) to run finetuned models and baselines
- llama.py to run LLama 3
- experiment_llm.ipynb to run gpt-4o-mini and parse outputs of LLMs

Additionnaly:
- postprocessing_ClimaQA_MRR.ipynb is used to compute the MRR performance on climaQA.
- climateFEVER.ipynb is used to compute the performance on climateFEVER aggregated and the human performances
- experiment_lobbymap.ipynb is used to compute the performance using the custom metrics of LobbyMap

Feel free ton contact the authors if you have any questions