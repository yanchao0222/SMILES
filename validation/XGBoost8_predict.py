import pandas as pd
import xgboost as xgb
import numpy as np
import math
import pickle

import os

""" gpu """
gpu_id = []
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_id)

evaluate_result = []
lab_name = ['PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB', 'MCV', 'PLT', 'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU']

NORM = np.array([103.60919172, 4.09204719,25.50716671,138.56716363,30.08871304,10.04918433,90.30916338,250.02270207,11.32065882,16.22554337,32.77929399,1.6120934,130.27666338])
STD = np.array([6.14830006,0.60043011,5.00476506,4.94450959,4.76858583,1.63745679,6.56035119,166.09335688,7.55992873,2.46055167,25.19313938,1.64061308,59.0473318])


def Test(model, X_test):
    # from train_with_missing
    # predict missing value for validation set
    print(model.predict(X_test))


    # plot_importance(model)
    # plt.show()

    return model.predict(X_test)


def each_ID_testData(id, lab):
    XList = []
    YList = []
    maxList = []
    minList = []

    filled_path = '../mix_impute/' + str(id) + ".csv"
    data_path = '../train_groundtruth/' + str(id) + ".csv"

    filled = pd.read_csv(filled_path)
    data = pd.read_csv(data_path)

    data_num = len(filled)

    time = data['CHARTTIME']

    for row in range(0, data_num):
        row_filled_front = row - 1
        row_filled_after = row + 1
        # TODO add more rows features
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

        tmp_list = []

        # TODO add more rows features

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

    return XList


if __name__ == '__main__':

    model = []
    model8 = pickle.load(open("model8.pickle.dat", "rb"))
    model.append(model8)

    # TODO change test size
    for id in range(7445, 8268):

        miss_path = '../labeled/new' + str(id) + ".csv"
        dataset = pd.read_csv(miss_path, header=None, skiprows=1);

        # TODO change col model
        for col in range(8, 9):  # to skip Charttime

            x_id_test = each_ID_testData(id, lab_name[col - 1])
            result = Test(model[0], x_id_test)

            s = pd.Series(dataset[col])

            nan_index_col = [index for index in range(len(s)) if s[index] == 'MISS']

            for i in range(len(result)):
                if (i in nan_index_col):
                    lab = (i, col)
                    thistuple = (id, lab, float(result[i]))
                    evaluate_result.append(thistuple)

print(evaluate_result)
np.save("output8", evaluate_result)
