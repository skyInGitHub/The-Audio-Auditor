import os
import glob
import pandas as pd
import argparse
import numpy as np
from random import sample
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt


def avg_results_user(n_sample):
    # os.chdir(chdir)
    # extension = 'csv'
    all_filenames = [i for i in glob.glob("results/DT_report_user/user{}_*.csv".format(n_sample))]

    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1score = 0
    # avg_sup = 0

    for f in all_filenames:
        result_csv = pd.read_csv(f)

        if result_csv.iloc[1, 0] == 'member':
            sum_accuracy += result_csv.iloc[1, 5]
            sum_precision += result_csv.iloc[1, 2]
            sum_recall += result_csv.iloc[1, 3]
            sum_f1score += result_csv.iloc[1, 1]
        else:
            print("The 2nd row is not 'member' row.")
            exit(1)

    avg_accuracy = sum_accuracy / len(all_filenames)
    avg_precision = sum_precision / len(all_filenames)
    avg_recall = sum_recall / len(all_filenames)
    avg_f1score = sum_f1score / len(all_filenames)

    return avg_accuracy, avg_precision, avg_recall, avg_f1score


def avg_results_fea(n_sample):
    # os.chdir(chdir)
    # extension = 'csv'
    all_filenames = [i for i in glob.glob('results/DT_report_fea/fea{}_*.csv'.format(n_sample))]

    sum_accuracy = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1score = 0
    # avg_sup = 0

    for f in all_filenames:
        result_csv = pd.read_csv(f)

        if result_csv.iloc[1, 0] == 'member':
            sum_accuracy += result_csv.iloc[1, 5]
            sum_precision += result_csv.iloc[1, 2]
            sum_recall += result_csv.iloc[1, 3]
            sum_f1score += result_csv.iloc[1, 1]
        else:
            print("The 2nd row is not 'member' row.")
            exit(1)

    avg_accuracy = sum_accuracy / len(all_filenames)
    avg_precision = sum_precision / len(all_filenames)
    avg_recall = sum_recall / len(all_filenames)
    avg_f1score = sum_f1score / len(all_filenames)

    return avg_accuracy, avg_precision, avg_recall, avg_f1score


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_start', type=int, help='start number of sample')
    parser.add_argument('n_end', type=int, help='end number of sample')
    parser.add_argument('n_step', type=int, help='end number of sample')
    # parser.add_argument('n_time', type=int, help='nth time for average result.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    # time = [0, 1, 2, 3]
    # position = [0, 100, 200, 300]
    #
    # plt.plot(time, position)
    # plt.xlabel('Time (hr)')
    # plt.ylabel('Position (km)')

    # for different features number
    args = get_arguments()
    n_start = args.n_start
    n_end = args.n_end
    n_step = args.n_step
    # n_start = 1
    # n_end = 9
    # n_step = 1

    train_csv = "data/train/train.csv"
    test_csv = "data/test/test.csv"
    avg_csv = "results/DT_report_fea/avg_fea_{}_{}.csv".format(n_start, n_end)      # for different features number
    # avg_csv = "results/DT_report_user/avg_user_{}_{}.csv".format(n_start, n_end)  # for different training size
    # avg_csv = "results/DT_report_user/avg_user_10_50_150.csv"                     # for different training size

    # n_time = 1
    # result_csv = "results/DT_report_fea/fea{}_{}.csv".format(n_sample, n_time)
    # result_txt = "results/DT_report_user/user{}_{}.csv".format(n_sample, n_time)

    # header = ['1_Feature', '2_Feature', '3_Feature', '4_Feature', '5_Feature', '6_Feature', '7_Feature', '8_Feature']
    header = ['accuracy', 'precision', 'recall', 'f1-score']
    result_fig = pd.DataFrame(columns=header)

    # user_number = [10, 20, 30, 40, 50, 75, 100, 150]      # for different training size

    # for n_sample in user_number:                          # for different training size
    for n_sample in range(n_start, n_end, n_step):          # for different features number
        avg_accuracy, avg_precision, avg_recall, avg_f1score = avg_results_fea(n_sample)        # for different features number
        # avg_accuracy, avg_precision, avg_recall, avg_f1score = avg_results_user(n_sample)     # for different training size

        result_fig.loc[n_sample, 'accuracy'] = avg_accuracy
        result_fig.loc[n_sample, 'precision'] = avg_precision
        result_fig.loc[n_sample, 'recall'] = avg_recall
        result_fig.loc[n_sample, 'f1-score'] = avg_f1score

    result_fig.to_csv(avg_csv)

    # print("The average results are:\n Average accuracy: {};\t Average precision: {};\t Average recall: {};\t Average"
    #       " F1-score: {}.".format(avg_accuracy, avg_precision, avg_recall, avg_f1score))
    print("The statistic result is located at: {}.".format(avg_csv))







