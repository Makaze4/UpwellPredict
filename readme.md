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
* pipeline.jpg: Diagram illustrating the structure of the workflow pipeline.

### Folders:
* input_data: Contains the input data for the pipeline, including grib and mat files (divided into North and South).
* experimentsInput: Contains the datasets for testing the pipeline.
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
* Before the actual execution the user is prompted to input:
  * Year (2004-2019)
  * Region ('N'-North or 'S'-South)
  * Dataset (ds1, ds2, ds3)

---

## Sample data from the years 2016 and 2018: North and South Morocco-experimentsInput

---

* The input data folder is divided into 2 folders: North and South
  * In each folder the following files are present:
    * Auxiliary *.grib and *.mat files for the preprocessing steps 
    * Folders of the corresponding years containing the *.grib file from the full year extracted from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS) and 
    the *.mat files containing the original SST information of each week

## Output data-experimentsOutput

---
Also divided into North and South, the output data contains:
  * Folders containing the files obtained from the preprocessing pipeline (daily_averages, rotated_data, zoomed_data)
  * Preprocessing pipeline visualization results of the first 8 days (inside the folder plots)
  * Trapezoidal membership functions obtained in the corresponding year (inside the folder plots)
  * Under the specified feature collection in the beggining:
    * Excel (.xlsx) file containing the optimal model's parameters and train-validation and test set classification results
    * Excel (.xlsx) file containing the optimal model's rule set
    * The files are organized according to the depths tested in the pipeline (3 to 10)
  * Random Forest feature importances (if executed)
  




## Note

The software as of testing is working, however it should be noticed that some results may differ
from the ones presented in the paper due to the random nature of the machine learning algorithms used.
The results are reproducible, but the optimal parameters
and subsequent rule sets may vary slightly due to the random nature of the algorithms.