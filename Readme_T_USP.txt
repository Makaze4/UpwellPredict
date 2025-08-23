# T-USP — Tree-based Upwelling Stability Periods Classification System

T-USP  (Tree-based Upwelling Stability Periods Classification System) is a framework for classifying  "Upwelling Stability Periods" (USPs) from daily wind-stress anomaly (WSA) data using tree-based machine learning models.  

T-USP extends the "Core–Shell clustering" methodology — originally proposed for USP detection from SST fields (see [Nascimento et al., *Computers & Geosciences*, 2023, https://doi.org/10.1016/j.cageo.2023.105421]) — by providing a reproducible pipeline that maps USP intervals onto daily scales and applies tree-based classification techniques.


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




