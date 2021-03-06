# XGBoost Missing Lab Tests Imputation

*******************************************************
**IEEE ICHI Data Analytics Challenge on Missing data Imputation (DACMI)**
*******************************************************
>***Team members***:

>**Xinmeng Zhang**, EECS, Vanderbilt University

>**Chao Yan**,  EECS, Vanderbilt University

>**Cheng Gao**, DBMI, Vanderbilt University Medical Center

>***Advised by***: 

>**Bradley Malin**,  DBMI,  Vanderbilt University Medical Center

>**You Chen**, DBMI,  Vanderbilt University Medical Center



<br />
<br />

This method utilizes eXtreme Gradient Boosting (**XGBoost**) as one of our methods of imputing missing values in patients' lab tests.

## XGBoost Framework
Achieving the missing value imputation by our imputor consists of three main parts: **1) prefill**, **2) train XGBoost based models**, and **3) apply**.

- Prefilled csv files are in the `mix_impute` folder for training set data and `test_mix_impute` folder for testing data. They support model training procedure.
- `labeled` folder contains 8267 training phase csv files that label original missing and artificially masked missing.
- `challenge_test_data` folder has 8267 csv files for testing phase.
- Each lab test would be imputed with their corresponding models.   `evaluate_imp_performance.py` and `lab_data.npy` are for imputation evaluation. Place `output.npy` generated by files in `validation` in the same folder as     `evaluate_imp_performance.py`.
- Make sure to install `pandas`, `xgboost`, `numpy`, `math`, `os` and `pickle` packages prior to running the code.
- Please refer to code comment blocks for implementation details. Since`train` and `validation` each has 13 `py` files for 13 laboratory tests, only the files for the first test is commented (`XGBoost1.py` & `XGBoost1_predict.py`), and the rest have same implementations. 


### Prefill
There are a wide variety of methods to address the missing data problem. In this data imputation challenge, we tested multiple existing methods to prefill the missing data. We utlized methods including global mean (mean of single variable across all patients), local mean (mean of neighboring values with window size 3), iterative SVD (iterative algorithm for estimating the singular value decomposition) and Soft-Impute (iteratively replaces the missing elements with the values obtained from a soft-thresholded SVD). By evalutating the imputation results on validation data, we were able to reach a conclusion that different lab tests can be better imputed with different imputing methods. Specifically, filling with local mean had better imputing performance with labs, including PCL, PLCO2, MCV, PLT, WBC, RDW, PBUN and PCRE while the rest labs such as PK, PNA, HCT, HGB, and PGLU were better imputed using matrix completion (soft impute). We utilize local mean and Soft-Impute strategies, which are more effective than the other two in prefilling training data and test data.

By running `prefill.py`, all the missings in folder `train_with_missing/` can be prefilled and then saved into folder `mix_impute/`. This is done and unnecessary to rerun. If needed, make sure to successfuly install Python package `fancyimpute`.


### Train XGBoost regressor
 The training model source code is in the `train` folder, supporting training models for each individual lab test. We trained 13 models, one for each laboratory test, and reached optimal performances. Using model for 'PCL' test for an example, `XGBoost1.py` fits the model with given optimal parameters and outputs the best performancing models. It takes files from `../mix_impute/` as prefill data and files from `../train_groundtruth/` to indicate missing value positions. 

Please refer to our paper for the details of the model training and the performance results.


### Predict missing data and evaluate result

We evaluated performances on the validation set using models with different parameters for tree booster (n_estimator, max_depth and min_child_weight). 13 models are in `model` folder. 

The files in `validation` folder load the models and predict missing value for validation set. They will output `output.npy`, which are lists of tuples, representing imputation value and position. They take files from `../mix_impute/` as prefill data and files from `../train_groundtruth/` to indicate missing value positions. They also load files from `labeled/` for training phase evaluation purpose.

Using model for 'PCL' test for an example, `XGBoost1_predict.py` loads the model from `model1.pickel.dat` file, predicts the missing values and saves the result to `output1.npy` file for evaluation. 


### Apply to testing data
`test_fill.py` loads the models with best performances and predicts missing values for testing data. The imputed csv files are in the `XGB_test_fill_result` folder.