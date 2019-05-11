def func_eval_imp_perf(filled_value, metric = 'RMSE range norm'):
    # This function is used to evaluate imputation performance
    # input: filled_value is an array and each element in this array represents one imputation
    # an example of this element is (1000, (2, 5), 35.4), the first item (1000) is the index of patient, the second item
    # ((2, 5)) is location of imputed lab (the fifth lab at the second timestamp).
    # the last item is the imputed value.

    import numpy as np
    import math

    lab_data_all_to_be_filled, gt_data_all, lab_data_len, gt_data_len, need_be_filled_all = np.load('lab_data.npy')

    # metric = 'RMSE std norm'
    # metric = 'RMSE range norm'
    # metric = 'MAPE'
    if metric == 'RMSE std norm':
        all_gt_data_mat = np.empty((0, gt_data_all[0].shape[1]), dtype=object)
        for idx in range(len(gt_data_all)):
            all_gt_data_mat = np.concatenate((all_gt_data_mat, gt_data_all[idx]), axis=0)

        # store standard deviation for labs
        htsd = []
        for idx_lab in range(gt_data_all[0].shape[1]):
            lab = [a for a in all_gt_data_mat[:, idx_lab] if not math.isnan(a)]
            htsd.append(np.std(lab))

    # start evaluate filled values
    error = np.zeros((gt_data_all[1].shape[1]))
    number_missing = np.zeros((gt_data_all[1].shape[1]))
    for idx in range(len(filled_value_sample)):
        idx_pt, loc, eimp = filled_value_sample[idx][0], filled_value_sample[idx][1], filled_value_sample[idx][2]
        print(idx_pt)
        print(loc)
        eraw = gt_data_all[idx_pt].iloc[loc[0], loc[1]]
        # print('Lab: ' + str(loc[1]))
        print([eimp, eraw])
        number_missing[loc[1]] += 1
        if metric == 'RMSE std norm':
            # print('Lab sd: ' + str(htsd[loc[1]]))
            # print(((eimp - eraw) / htsd[loc[1]]) ** 2)
            error[loc[1]] += ((eimp - eraw) / htsd[loc[1]]) ** 2
        elif metric == 'RMSE range norm':
            lab = [a for a in gt_data_all[idx_pt].iloc[:, loc[1]] if not math.isnan(a)]
            # print(((eimp-eraw) / (np.max(lab) - np.min(lab))) ** 2)
            error[loc[1]] += ((eimp - eraw) / (np.max(lab) - np.min(lab))) ** 2
        elif metric == 'MAPE':
            # print(((eimp - eraw) / eraw) ** 2)
            if eraw != 0:
                error[loc[1]] += ((eimp - eraw) / eraw) ** 2
            # else:
            #     print([idx_pt, loc, eimp])
            #     break
    error[0] = np.sum(error[1:])
    number_missing[0] = np.sum(number_missing[1:])
    return([np.sqrt(error[idx] / number_missing[idx]) for idx in range(len(error))])

import numpy as np
import pandas as pd
filled_value_sample = np.load('output.npy')
perf_res = func_eval_imp_perf(filled_value_sample)
perf_res = pd.DataFrame(perf_res)
perf_res.to_csv('perf_res_max.csv', index=False)
print(func_eval_imp_perf(filled_value_sample))