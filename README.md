# BioNLP

Repository associated to submission to BioNLP 2023 Shared Task 1A. Problem List Summarization.
https://aclanthology.org/2023.bionlp-1.48/


## Software implementation

The repository is divided in 3 major sections in relation on the function each grup of the scripts carry out. The sections are the following:
  * Preprocessing
  * NER
  * Finetuning

Each of the sections have a folder named Resources which store the different CSV files for the scritps to load the data and generate the modified dataset.

### Preprocessing Data
The files used to analyse the data in order to perform then data cleaning including the de-identification and exploring the issues with abbreviations, this group of files is in the [Preprocessing folder](./Preprocessing/).

    .
    ├── ...
    ├── Preprocessing
    │   ├── BioAbbr.py                  # Recognizing and replacing abbreviation their full form phrase
    │   ├── BioNLP_Clean.py             # Removing noise and adapting the csv dataset
    │   ├── CheckWords.py               # Checking if the words in the ground truth are present in the data columns 
    │   ├── NER
    │   │   ├── ...
    │   └── DeIdentification.py         # Identifying the types of de-identification and the best way to replace them
    └── ...


### NER Labeling
This section has the code for adapting the dataset to Token classification, all inside of the [NER folder](./Preprocessing/NER/).

    .
    ├── ...
    │   ├── ...
    │   ├── NER
    │   │   ├── BioNLP_merge.py         # Modifying the NER labeled dataset with different column options
    │   │   ├── CheckLabels.py          # Manual checking of the labels for NER
    │   │   ├── NER_Labeling.py         # Using a pretrained model to automatically label the dataset
    │   │   ├── NER_UI.py               # Manual Labeling with UI
    │   │   └── NERdataT2S.py           # Creating a Dataset to train model to convert keywords to summary
    │   │   └── RougeDataset.py         # Testing the NER dataset with the Rogue metric
    │   └── ...
    └── ...


### Finetuning
The finetuning of the models and experiments code are on the Main folder having different files for the different variations

    .
    ├── BioBert.py                      # Finetuning BERT with biomedical data and MIMIC NER
    ├── BioBertCRF.py                   # Modified BERT with CRF
    ├── BioBertCustom.py                # Finetuning BERT using Data augmentation
    ├── BioBertCustomLoop.py            # Testing Data augmentation percentage
    ├── BioBertCustomLoss.py            # Finetuning BERT using a weigthed loss
    ├── BioBertRay.py                   # Finetuning BERT Hyperparameters
    ├── Pipeline.py                     # Script to generate predictions from input data both public and private tests
    ├── Preprocessing
    │   ├── ...
    │   ├── NER
    │   │   └── ...
    │   └── ...
    ├── SumWords.py                     # Evaluating Rouge on finetuned model concatenating the keywords
    ├── Summarization.py                # Finetuning T5 for text summarization
    └── WordsToText.py                  # Finetuning T5 for NER keywords to text summary


## Dependencies

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `environment.yml`.

We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `environment.yml`
is located) to create a separate environment and install all required
dependencies in it:

    conda env create
