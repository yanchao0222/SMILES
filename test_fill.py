
import pandas as pd
from pandas import read_csv
import numpy as np
import math
import pickle

import os

""" gpu """
gpu_id = ['0']
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_id)

#13 lab tests
lab_name = ['PCL', 'PK', 'PLCO2', 'PNA', 'HCT', 'HGB', 'MCV', 'PLT', 'WBC', 'RDW', 'PBUN', 'PCRE', 'PGLU']

#for normalization
NORM = np.array([103.60919172, 4.09204719,25.50716671,138.56716363,30.08871304,10.04918433,90.30916338,250.02270207,11.32065882,16.22554337,32.77929399,1.6120934,130.27666338])
STD = np.array([6.14830006,0.60043011,5.00476506,4.94450959,4.76858583,1.63745679,6.56035119,166.09335688,7.55992873,2.46055167,25.19313938,1.64061308,59.0473318])




#predict missing values
#return prediction result
def Test(model, X_test):
    # from train_with_missing
    # predict missing value for validation set
    print(model.predict(X_test))

    # show important characteristics
    # plot_importance(model)
    # plt.show()

    return model.predict(X_test)



#load validation model input for indicated patients and indicated lab test
#return input as a list
def each_ID_testData(id, lab):
    XList = []

    filled_path = 'test_mix_impute/' + str(id) + ".csv" #load prefill test data
    data_path = 'challenge_test_data/' + str(id) + ".csv" #load test data with missing values

    filled = pd.read_csv(filled_path)
    data = pd.read_csv(data_path)

    data_num = len(filled)

    time = data['CHARTTIME']

#load every row in the test data
    for row in range(0, data_num):
        row_filled_front = row - 1 #load front 2 rows and later 2 rows into input feature
        row_filled_after = row + 1
        # TODO add more rows features
        row_filled_front_front = row - 2
        row_filled_after_after = row + 2

        if (row == 0): #check if rows are outof range
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

    # load models for 13 lab tests
    model = []

    model1 = pickle.load(open("model/model1.pickle.dat", "rb"))
    model.append(model1)

    model2 = pickle.load(open("model/model2.pickle.dat", "rb"))
    model.append(model2)

    model3 = pickle.load(open("model/model3.pickle.dat", "rb"))
    model.append(model3)

    model4 = pickle.load(open("model/model4.pickle.dat", "rb"))
    model.append(model4)

    model5 = pickle.load(open("model/model5.pickle.dat", "rb"))
    model.append(model5)

    model6 = pickle.load(open("model/model6.pickle.dat", "rb"))
    model.append(model6)

    model7 = pickle.load(open("model/model7.pickle.dat", "rb"))
    model.append(model7)

    model8 = pickle.load(open("model/model8.pickle.dat", "rb"))
    model.append(model8)

    model9 = pickle.load(open("model/model9.pickle.dat", "rb"))
    model.append(model9)

    model10 = pickle.load(open("model/model10.pickle.dat", "rb"))
    model.append(model10)

    model11 = pickle.load(open("model/model11.pickle.dat", "rb"))
    model.append(model11)

    model12 = pickle.load(open("model/model12.pickle.dat", "rb"))
    model.append(model12)

    model13 = pickle.load(open("model/model13.pickle.dat", "rb"))
    model.append(model13)


#predict missing values and impute
    for id in range(1, 8268):

        #load test data into dataframe
        test_path = "challenge_test_data/" + str(id) + ".csv"
        data_test = read_csv(test_path);

        #fill test data column by column
        for col in range(1, 14):  # to skip ChartTime

            x_id_test = each_ID_testData(id, lab_name[col - 1]) #load model input values
            result = Test(model[col-1], x_id_test) #save model output values

            s = pd.Series(data_test[lab_name[col-1]]) #load test data dataframe into a series

            nan_index_col = [index for index in range(len(s)) if  np.isnan(s[index])]

            #set the imputed value to the dataframe
            for i in range(len(result)):
                if (i in nan_index_col):
                    value = float(result[i])
                    data_test.set_value(i, lab_name[col - 1], value)


        data_test.to_csv(r'XGB_test_fill_result/' + str(id) + ".csv", index=None, header=True)
        print(id, ' file finished')












