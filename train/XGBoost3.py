import pandas as pd
import xgboost as xgb
import numpy as np
import math
import pickle

import os

""" gpu """
gpu_id = ['0']
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_id)

NORM = np.array([103.60919172, 4.09204719,25.50716671,138.56716363,30.08871304,10.04918433,90.30916338,250.02270207,11.32065882,16.22554337,32.77929399,1.6120934,130.27666338])
STD = np.array([6.14830006,0.60043011,5.00476506,4.94450959,4.76858583,1.63745679,6.56035119,166.09335688,7.55992873,2.46055167,25.19313938,1.64061308,59.0473318])
lab_name = ['PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB', 'MCV', 'PLT', 'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU']


def colIndex(col):
    if col == 'PCL':
        return 1
    if col == 'PK':
        return 2
    if col == 'PLCO2':
        return 3
    if col == 'PNA':
        return 4
    if col == 'HCT':
        return 5
    if col == 'HGB':
        return 6
    if col == 'MCV':
        return 7
    if col == 'PLT':
        return 8
    if col == 'WBC':
        return 9
    if col == 'RDW':
        return 10
    if col == 'PBUN':
        return 11
    if col == 'PCRE':
        return 12
    if col == 'PGLU':
        return 13


evaluate_result = []


def featureSet(lab):
    XList = []
    yList = []
    maxList = []
    minList = []

    for id in range(1, 7445):
        filled_path = '../mix_impute/' + str(id) + ".csv"
        truth_path = '../train_groundtruth/' + str(id) + ".csv"

        filled = pd.read_csv(filled_path)
        truth = pd.read_csv(truth_path)


        data_num = len(filled)
        lab_name = ['PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB', 'MCV', 'PLT', 'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU']

        time = truth['CHARTTIME']

        for row in range(0, data_num):
            row_filled_front = row - 1
            row_filled_after = row + 1
            #TODO add more rows features
            row_filled_front_front = row-2
            row_filled_after_after = row+2

            if (row == 0):
                row_filled_front = row_filled_after
                row_filled_front_front = row_filled_after

            if (row == data_num - 1):
                row_filled_after = row_filled_front
                row_filled_after_after = row_filled_front

            ##TODO add more rows features
            if (row == 1):
                row_filled_front_front = row_filled_front

            if (row == data_num -2):
                row_filled_after_after = row_filled_after

            if ((not math.isnan(truth.iloc[row]['PCL'])) and
                    (not math.isnan(truth.iloc[row]['PK'])) and
                    (not math.isnan(truth.iloc[row]['PLCO2'])) and
                    (not math.isnan(truth.iloc[row]['PNA'])) and
                    (not math.isnan(truth.iloc[row]['HCT'])) and
                    (not math.isnan(truth.iloc[row]['HGB'])) and
                    (not math.isnan(truth.iloc[row]['MCV'])) and
                    (not math.isnan(truth.iloc[row]['PLT'])) and
                    (not math.isnan(truth.iloc[row]['WBC'])) and
                    (not math.isnan(truth.iloc[row]['RDW'])) and
                    (not math.isnan(truth.iloc[row]['PBUN'])) and
                    (not math.isnan(truth.iloc[row]['PCRE'])) and
                    (not math.isnan(truth.iloc[row]['PGLU']))):
                tmp_list = []

                #TODO add more rows features
                # append front front filled row
                for index in range(0, 13):
                    tmp_list.append( (filled.iloc[row_filled_front_front][lab_name[index]] - NORM[index]) / STD[index])

                # append front filled row
                for index in range(0, 13):
                    tmp_list.append( (filled.iloc[row_filled_front][lab_name[index]] - NORM[index]) / STD[index])

                # append truth row (except the col)
                for index in range(0, 13):
                    if (lab_name[index] != lab):
                        tmp_list.append( (truth.iloc[row][lab_name[index]] - NORM[index]) / STD[index])

                # append filled row after
                for index in range(0, 13):
                    tmp_list.append( (filled.iloc[row_filled_after][lab_name[index]] -  NORM[index]) / STD[index])

                # append filled row after after:
                for index in range(0, 13):
                    tmp_list.append( (filled.iloc[row_filled_after_after][lab_name[index]] - NORM[index]) / STD[index])

                time21 = (time[row]- time[row_filled_front])/60
                time32 = (time[row_filled_after] - time[row])/60
                time10 = (time[row_filled_front] - time[row_filled_front_front])/60
                time43 = (time[row_filled_after_after] - time[row_filled_after])/60

                tmp_list.append(time21)
                tmp_list.append(time32)
                tmp_list.append(time10)
                tmp_list.append(time43)

                XList.append(tmp_list)
                yList.append(truth.iloc[row][lab])

                min_num = np.nanmin(truth[lab])
                max_num = np.nanmax(truth[lab])

                maxList.append(max_num)
                minList.append(min_num)

    return XList, yList, maxList, minList


def loadTestData(lab):
    XList = []
    YList = []
    maxList = []
    minList = []

    lab_name = ['PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB', 'MCV', 'PLT', 'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU']

    # TODO change test size
    for id in range(7445, 8268):

        filled_path = '../mix_impute/' + str(id) + ".csv"
        data_path = '../train_groundtruth/' + str(id) + ".csv"

        filled = pd.read_csv(filled_path)
        data = pd.read_csv(data_path)

        data_num = len(filled)

        time = data['CHARTTIME']

        for row in range(0, data_num):
            row_filled_front = row - 1
            row_filled_after = row + 1

            #TODO add more rows features
            row_filled_front_front = row - 2
            row_filled_after_after = row + 2

            if (row == 0):
                row_filled_front = row_filled_after
                row_filled_front_front = row_filled_front

            if (row == data_num - 1):
                row_filled_after = row_filled_front
                row_filled_after_after = row_filled_front

            ##TODO add more rows features
            if (row == 1):
                row_filled_front_front = row_filled_front

            if (row == data_num - 2):
                row_filled_after_after = row_filled_after

            if ((not math.isnan(data.iloc[row]['PCL'])) and
                    (not math.isnan(data.iloc[row]['PK'])) and
                    (not math.isnan(data.iloc[row]['PLCO2'])) and
                    (not math.isnan(data.iloc[row]['PNA'])) and
                    (not math.isnan(data.iloc[row]['HCT'])) and
                    (not math.isnan(data.iloc[row]['HGB'])) and
                    (not math.isnan(data.iloc[row]['MCV'])) and
                    (not math.isnan(data.iloc[row]['PLT'])) and
                    (not math.isnan(data.iloc[row]['WBC'])) and
                    (not math.isnan(data.iloc[row]['RDW'])) and
                    (not math.isnan(data.iloc[row]['PBUN'])) and
                    (not math.isnan(data.iloc[row]['PCRE'])) and
                    (not math.isnan(data.iloc[row]['PGLU']))):

                tmp_list = []

            #TODO add more rows features

            # append front front filled row
                for index in range(0, 13):
                    tmp_list.append( (filled.iloc[row_filled_front_front][lab_name[index]] - NORM[index]) / STD[index])

            # append front filled row
                for index in range(0, 13):
                    tmp_list.append( (filled.iloc[row_filled_front][lab_name[index]] - NORM[index]) / STD[index])

            # append truth row (except the col)
                for index in range(0, 13):
                    if (lab_name[index] != lab):
                        tmp_list.append( (filled.iloc[row][lab_name[index]] - NORM[index]) / STD[index])

            # append filled row after
                for index in range(0, 13):
                    tmp_list.append( (filled.iloc[row_filled_after][lab_name[index]] - NORM[index]) / STD[index])

            # append filled row after after:
                for index in range(0, 13):
                    tmp_list.append( (filled.iloc[row_filled_after_after][lab_name[index]] - NORM[index]) / STD[index])


                time21 = (time[row] - time[row_filled_front])/60
                time32 = (time[row_filled_after] - time[row])/60
                time10 = (time[row_filled_front] - time[row_filled_front_front])/60
                time43 = (time[row_filled_after_after] - time[row_filled_after])/60

                tmp_list.append(time21)
                tmp_list.append(time32)
                tmp_list.append(time10)
                tmp_list.append(time43)

                XList.append(tmp_list)
                YList.append(data.iloc[row][lab])

                min_num = np.nanmin(filled[lab])
                max_num = np.nanmax(filled[lab])

                maxList.append(max_num)
                minList.append(min_num)

    return XList, YList, maxList, minList


def train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small):
    # all from groundtruth
    def nRMSE(pred, y):

        max_num = np.array(max_test)
        min_num = np.array(min_test)

        if (len(pred) != len(max_num)):
            max_num = np.array(max_small)
            min_num = np.array(min_small)
            result = np.mean(((pred - y.get_label()) ** 2) / (max_num - min_num))

        else:
            result = np.mean(((pred - y.get_label()) ** 2) / (max_num - min_num))

        return ('nRMSE', result)

    # TODO parameter adjust
    # XGBoost Training Process
    eval_set = [(X_train, y_train), (X_test, y_test)]
    # eval_set = [(X_train, y_train)]
    other_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 5, 'min_child_weight': 1}

    model = xgb.XGBRegressor(**other_params)

    model.fit(X_train, y_train, early_stopping_rounds=20, eval_set=eval_set, eval_metric=nRMSE)
    # model.fit(X_train, y_train,early_stopping_rounds=50,eval_set=eval_set)
    # early stop change to 20/30

    # save model to file
    pickle.dump(model, open("model3.pickle.dat", "wb"))

    return model



if __name__ == '__main__':

    model = []

    # labName1 = 'PCL'
    # X_train, y_train, max_test, min_test = featureSet(labName1)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName1)  # test data list
    # model1 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model1)
    #
    # # TODO change lab model
    # labName2 = 'PK'
    # X_train, y_train, max_test, min_test = featureSet(labName2)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName2)  # test data list
    # model2 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model2)

    labName3 = 'PLCO2'
    X_train, y_train, max_test, min_test = featureSet(labName3)  # train data list
    X_test, y_test, max_small, min_small = loadTestData(labName3)  # test data list
    model3 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    model.append(model3)

    # labName4 = 'PNA'
    # X_train, y_train, max_test, min_test = featureSet(labName4)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName4)  # test data list
    # model4 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model4)
    #
    # labName5 = 'HCT'
    # X_train, y_train, max_test, min_test = featureSet(labName5)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName5)  # test data list
    # model5 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model5)
    #
    # labName6 = 'HGB'
    # X_train, y_train, max_test, min_test = featureSet(labName6)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName6)  # test data list
    # model6 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model6)
    #
    # labName7 = 'MCV'
    # X_train, y_train, max_test, min_test = featureSet(labName7)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName7)  # test data list
    # model7 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model7)
    #
    # labName8 = 'PLT'
    # X_train, y_train, max_test, min_test = featureSet(labName8)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName8)  # test data list
    # model8 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model8)
    #
    # labName9 = 'WBC'
    # X_train, y_train, max_test, min_test = featureSet(labName9)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName9)  # test data list
    # model9 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model9)
    #
    # labName10 = 'RDW'
    # X_train, y_train, max_test, min_test = featureSet(labName10)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName10)  # test data list
    # model10 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model10)
    #
    # labName11 = 'PBUN'
    # X_train, y_train, max_test, min_test = featureSet(labName11)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName11)  # test data list
    # model11 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model11)
    #
    # labName12 = 'PCRE'
    # X_train, y_train, max_test, min_test = featureSet(labName12)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName12)  # test data list
    # model12 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model12)
    #
    # labName13 = 'PGLU'
    # X_train, y_train, max_test, min_test = featureSet(labName13)  # train data list
    # X_test, y_test, max_small, min_small = loadTestData(labName13)  # test data list
    # model13 = train(X_train, y_train, X_test, y_test, max_test, min_test, max_small, min_small)
    # model.append(model13)



