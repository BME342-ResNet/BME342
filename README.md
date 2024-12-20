# Supplementary material

## Installation instructions

1. Setup the environment by creating a new conda environment from the definition file: ```conda env create --name <choose_name> --file=environment.yml```
2. Make sure the dataset is placed in the project root folder and named ```IMC_Images```. The supplementary metadata ```panel.csv``` and ```metadata.csv``` should be placed in the project root folder. The dataset is not being redistributed.

## Model Execution

- The project is split into different files, based on the stage of the experiment and the experiment performed.

- ```code/DataPreparation.ipynb``` contains code related to EDA of the dataset.
- ```code/Model Architecture/ResNet``` contains different notebooks with models that were used as tests and for the final model.