import numpy as np
import pandas as pd
import math
from os import listdir
from fancyimpute import SoftImpute
from os.path import isfile, join
from copy import deepcopy

# data path: in this data path, files names such as 1.csv, 2.csv and 3.csv are stored
train_data_path = './challenge_test_data/'
# the path used to store mix imputation result
mix_impute_path = './test_mix_impute/'
# all the csv file stored in file_names
file_names = [f for f in listdir(train_data_path) if isfile(join(train_data_path, f))]
# lab data stored in lab_data_all, which is a dictionary. Key is the names of csv files, starting from 1.
lab_data_all = {}
for fn in file_names:
    # lab data
    idx_pt = int(fn.split('.')[0])
    lab_data = pd.read_csv(train_data_path + fn)
    lab_data_all[idx_pt] = lab_data

# start filling lab_data with two basic methods, mean imputation and matrix completion
# One of the basic imputation methods: mean imputation.
# For the lab data needed to be imputed, look up (the previous time points) for the first non-NA (called A) and look
# down (the later time points) for the first non-NA (called B). The imputed value is the mean of these two values (A
# and B).
def mean_impute(lab_data, na_loc):
    '''This function performs imputation with mean. Two inputs lab_data and na_loc; output imputed value lab_data stores lab data for one patient and na_loc stores the location of missing, at which lab and which timestamp lab is missing.'''
    # used to store the up value
    up_cand_val = []
    # search up
    start = na_loc[0]
    while start - 1 >= 0:
        if  not math.isnan(lab_data.iloc[start-1, na_loc[1]]):
            up_cand_val.append(lab_data.iloc[start-1, na_loc[1]])
        start = start - 1
        if len(up_cand_val) == 1:
            break
    # used to store the down value
    down_cand_val = []
    # search down
    start = na_loc[0]
    while start + 1 < int(len(lab_data)):
        if not math.isnan(lab_data.iloc[start + 1, na_loc[1]]):
            down_cand_val.append(lab_data.iloc[start + 1, na_loc[1]])
        start = start + 1
        if len(down_cand_val) == 1:
            break
    # take the mean to impute
    cand_val = up_cand_val + down_cand_val
    return(np.mean(cand_val))

# prepare for softimpute, one of the methods using matrix completion. Normalize lab data
def normalize_0_1(lab_data):
    '''This function is used to normalize lab data before matrix completion. We normalize each lab data for each patient. Input lab_data stores lab data for one patient. Output normalized lab_data with the same dimensions as lab_data, mean and standard deviation, which are used to calculate imputed value after performing matrix completion'''
    # make a hard copy of lab data
    lab_data_n = deepcopy(lab_data)
    # store mean and standard deviation for calculating the imputed value after performing matrix completion
    lab_mean_all, lab_std_all = [], []
    for idx_colmn in range(1, lab_data.shape[1]):
        lab = [a for a in lab_data.iloc[:, idx_colmn]]
        lab_exclude_nan = [a for a in lab if not math.isnan(a)]
        lab_mean, lab_std = np.mean(lab_exclude_nan), np.std(lab_exclude_nan)
        lab_data_n.iloc[:,idx_colmn] = [(a-lab_mean) / lab_std if not math.isnan(a) else a
                                        for a in lab]
        lab_mean_all.append(lab_mean)
        lab_std_all.append(lab_std)
    return(lab_data_n, lab_mean_all, lab_std_all)

# these labs need to be imputed using simple mean method. These index was determined using training data, and simple
# mean method resulted best imputation performance. Other labs were best imputed by matrix completion.
need_mean_impute_index = [1, 3, 7, 8, 9, 10, 11, 12]
# need_soft_impute_index = [2, 4, 5, 6, 13]
for idx_pt in range(1, len(lab_data_all) + 1):
    lab_data = deepcopy(lab_data_all[idx_pt])
    rows, cols = np.where(pd.isna(lab_data))
    lab_na_loc = [(rows[a], cols[a]) for a in range(len(rows))]

    # prepare for softimpute (matrix completion)
    # normalize
    lab_data_n, lab_mean_all, lab_std_all = normalize_0_1(lab_data)
    # soft impute
    res = SoftImpute(max_iters=1000, verbose=False).fit_transform(lab_data_n.iloc[:, 1:])

    for na_loc in lab_na_loc:
        if na_loc[1] in need_mean_impute_index:
            # simple mean impute
            lab_data.iloc[na_loc[0], na_loc[1]] = mean_impute(lab_data, na_loc)
        else:
            # calculate the imputed value using mean and standard deviation
            imp_val = res[na_loc[0], na_loc[1] - 1] * lab_std_all[na_loc[1] - 1] + lab_mean_all[na_loc[1] - 1]
            lab_data.iloc[na_loc[0], na_loc[1]] = imp_val
    # save imputed csv to mix impute path
    lab_data.to_csv(mix_impute_path + str(idx_pt) + '.csv', index=False)
print('Prefill completed!')


