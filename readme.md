# T-USP — Tree-based Upwelling Stability Periods Classification System

* T-USP  (Tree-based Upwelling Stability Periods Classification System) is a framework for classifying  
"Upwelling Stability Periods" (USPs) from daily wind-stress anomaly (WSA) data using tree-based machine learning models.

* T-USP extends the "Core–Shell clustering" methodology — originally proposed for USP detection from SST fields 
(see [Nascimento et al., *Computers & Geosciences*, 2023, https://doi.org/10.1016/j.cageo.2023.105421]) — 
by providing a reproducible pipeline that maps USP intervals onto daily scales and applies tree-based classification techniques.




## Features 
The  T-USP pipeline follows the workflow described in the accompanying paper and implements:

1. **Preprocessing of ERA5 winds**
   - Data cleaning, temporal aggregation, alongshore wind computation, spatial resampling
   - Wind stress anomaly (WSA) map computation

2. **USP Label Assignment to WSA maps**
    - Mapping Core–Shell derived USP intervals from weekly SST instants to daily scale
    - Time-series fuzzification and defuzzification (MOM / SoM)
    - Production of daily labeled WSA maps

3. **Feature Set Extraction and Dataset Construction**
    - Zonal division of WSA/SST maps
    - Extraction of per-zone statistics (Min, Avg, Max)
    - Coastal–offshore SST difference features
    - Configurable datasets:
        - **DS1** (WSA only)
        - **DS2** (WSA + SST differences)
        - **DS3** (WSA + SST statistics + SST differences)

4. **Tree-based Classification Models**
    - Training and validation of Decision Tree and Random Forest classifiers
    - Model selection and performance evaluation on independent test sets
    - Extraction of interpretable rule sets and feature importance

5. **Integration with Core–Shell Clustering**
    - Seamless extension of the Core–Shell framework:  
      Core–Shell → USP time intervals (from SST) → **T-USP** → classification of daily WSA maps into USP classes





## Repository Overview

* *.grib files are used as the Wind maps, with two auxiliary *.grib files containing SST information. *.mat files are used as SST grids.
* In this repository is present the fully proposed pipeline, including the preprocessing steps, USP label assignment, feature extraction and model building and evaluation, along with
  some auxiliary functions for logging and plotting the results.

---
### Files:

* pipelineScript.py: Main script for running the UpwellPredict pipeline.
* preprocesser.py: Implementation of the preprocessing pipeline steps.
* feature_extractor.py: Implementation of the feature extraction steps.
* dataset_builder.py: Classifies the extracted feature set.
* model_building.py: Implementation of the model training and evaluation steps with decision trees and random forests.
* loggers.py: Auxiliary functions for logging the pipeline's progress and results.
* plotters.py: Auxiliary functions for plotting the results of the pipeline.
* requirements.txt: List of required Python packages.
* User_Instructions.txt: Instructions for using the UpwellPredict pipeline.
* pipeline.jpg: Diagram illustrating the structure of the workflow pipeline.

### Folders:
* input_data: Contains the input data for the pipeline, including grib and mat files (divided into North and South).
* experimentsInput: Contains the datasets for the model building and evaluation phase of the pipeline.
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

* Download the latest version from the repository and place it in a directory of your choice.
* It is recommended to use an IDE to run the software, as it uses several paths as inputs to the program.
* To run the software, execute the pipelineScript.py file.
* Before the actual execution the user is prompted to input:
  * Year (2004-2019)
  * Region ('N'-North or 'S'-South)
  * Dataset (ds1, ds2, ds3)

---

## Sample data from the years 2016 and 2018: North and South Morocco-input_data

---

* The input data folder is divided into 2 folders: North and South
  * In each folder the following files are present:
    * Auxiliary *.grib and *.mat files for the preprocessing steps 
    * Folders of the corresponding years containing the *.grib file from the full year extracted from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS) and 
    the *.mat files containing the original SST information of each week.

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
  * Random Forest feature importance's (if executed)
  




## Notes

---

* The software as of testing is working, however it should be noticed that some results may differ
from the ones presented in the paper due to the random nature of the machine learning algorithms used.
The results are reproducible, but the optimal hyperparameters
and subsequent rule sets may vary slightly due to the random nature of the algorithms.
* User Warnings are issued but these can be ignored as they do not affect the results of the pipeline.