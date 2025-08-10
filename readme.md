# Upwelling Prediction Pipeline

This repository contains a Python based pipeline for the prediction of upwelling stability periods using machine learning techniques.

## Repository Overview

---
### Files:

* pipelineScript.py: Main script for running the upwelling prediction pipeline.
* preprocesser.py: Implementation of the preprocessing pipeline steps.
* feature_extractor.py: Implementation of the feature extraction steps.
* dataset_builder.py: Classifies the extraced features dataset
* model_building.py: Implementation of the model training and evaluation steps with decision trees and random forests.
* loggers.py: Functions for logging the pipeline's progress and results.
* plotters.py: Functions for plotting the results of the pipeline.
* requirements.txt: List of required Python packages.

### Folders:
* input_data: Contains the input data for the pipeline, including grib and mat files (divided into North and South).
* experimentsInput: Contains pre-made datasets for testing the pipeline.
* experimentsOutput: Contains the output data from the pipeline, including Excel files with the results (divided into North and South).

## Software Requirements

---

* Python 3.9 or higher
* In case of missing libraries, install them using pip:
```bash
pip install -r requirements.txt
```

## Run the software

---

* Download the repository and place it in a directory of your choice.
* It is recommended to use an IDE to run the software, as it uses several paths as inputs to the program
* To run the software, execute the pipelineScript.py file.

Note: 
* The software as of testing is working, it is recommended to run only the third stage of the pipeline due to the final step of the preprocessing pipeline generating huge amounts of data.
* For this matter, 2 pre-made datasets are available in the folder 'experimentsInput'

## Sample data from the year 2019: North and South Morocco

---

* A sample of the full dataset is present in the folder 'input_data'
* The input data is divided into 2 folders: North and South
  * In each folder the following files are present:
    * The *.grib file from the full year extracted from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS)
    * The *.mat files containing the original SST information of each week
    * An additional auxiliary *.grib and *.mat files to be used during the preprocessing steps

## Output data

---

* Excel (.xlsx) file containing the optimal model's parameters and train-validation and test set classification results
* Excel (.xlsx) file containing the optimal model's rule set
* The files are organized according to the depths tested in the pipeline (3 to 10)
* Preprocesing pipeline resutls of the first 8 days
* Trapezoidal membership functions obtained in the corresponding year