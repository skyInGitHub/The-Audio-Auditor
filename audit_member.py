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


def svm(X_train, X_test, y_train, y_test, confu_csv, result_txt):
    # Transform the str type in dataset to value so that can be trained with fit() function
    X_train = label_encoder(X_train)
    X_test = label_encoder(X_test)

    classifier = SVC(gamma='auto')
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    confu_matrix = confusion_matrix(y_test, y_pred)
    pd.DataFrame(confu_matrix).to_csv(confu_csv)

    result_report = classification_report(y_test, y_pred, output_dict=True)
    result_report = pd.DataFrame(result_report).transpose()

    accuracy = accuracy_score(y_test, y_pred)
    result_report['Accuracy'] = accuracy
    with open(result_txt, 'w') as f:
        f.write(result_report)

    return confu_matrix, result_report, accuracy


def decision_tree(X_train, X_test, y_train, y_test, confu_csv, result_txt):
    # Transform the str type in dataset to value so that can be trained with fit() function
    X_train = label_encoder(X_train)
    X_test = label_encoder(X_test)

    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    confu_matrix = confusion_matrix(y_test, y_pred)
    pd.DataFrame(confu_matrix).to_csv(confu_csv)

    result_report = classification_report(y_test, y_pred, output_dict=True)
    result_report = pd.DataFrame(result_report).transpose()

    accuracy = accuracy_score(y_test, y_pred)
    result_report['Accuracy'] = accuracy
    result_report.to_csv(result_txt)
    # result_report
    # with open(result_txt, 'w') as f:
    #     f.write(result_report)

    return confu_matrix, result_report, accuracy


def label_encoder(train_data):
    le = preprocessing.LabelEncoder()

    for column_name in train_data.columns:
        if train_data[column_name].dtype == object:
            train_data[column_name] = le.fit_transform(train_data[column_name])
        else:
            pass

    return train_data


def random_train(X_train, y_train, n_sample):

    train_len = len(X_train)
    train_indices = sample(range(train_len), n_sample)

    X_train = X_train.iloc[train_indices]
    y_train = y_train.iloc[train_indices]

    return X_train, y_train


def random_test_feature(X_test, n_sentence):

    test_fea = [0, 3, 6, 9, 12, 15, 18, 21]
    indices = sample(test_fea, n_sentence)
    test_indices = []
    for i in indices:
        test_indices.append(int(i))
        test_indices.append(int(i+1))
        test_indices.append(int(i+2))

    # X_test = X_test.iloc[test_indices]

    # test_not = []
    for i in range(24):
        if i not in test_indices:
            # test_not.append(i)
            X_test.iloc[:, i] = -1

    return X_test


def avg_results(chdir, ):
    os.chdir(chdir)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    avg_pre = 0
    avg_recall = 0
    avg_f1 = 0
    avg_sup = 0

    for f in all_filenames:
        result_csv = pd.read_csv(f)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('n_sample', type=int, help='the amount number of random users/features')
    parser.add_argument('n_time', type=int, help='nth time for average result.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':

    train_csv = "data/train/train.csv"
    test_csv = "data/test/test.csv"
    # train_des_csv = "data/train/train_description.csv"
    # test_des_csv = "data/test/test_description.csv"

    args = get_arguments()
    n_sample = args.n_sample
    n_time = args.n_time

    confu_csv = "results/DT_confusion_fea/fea{}_{}.csv".format(n_sample, n_time)
    result_txt = "results/DT_report_fea/fea{}_{}.csv".format(n_sample, n_time)
    # confu_csv = "results/DT_confusion_user/user{}_{}.csv".format(n_sample, n_time)
    # result_txt = "results/DT_report_user/user{}_{}.csv".format(n_sample, n_time)
    # confu_csv = "results/SVM_confusion_matrix.csv"
    # result_txt = "results/SVM_report.csv"

    train_set = pd.read_csv(train_csv)
    test_set = pd.read_csv(test_csv)

    # Save dataset description
    # train_des = train_set.describe()
    # test_des = test_set.describe()
    # pd.DataFrame(train_des).to_csv(train_des_csv)
    # pd.DataFrame(test_des).to_csv(test_des_csv)

    X_train = train_set.drop('class', axis=1)
    y_train = train_set['class']
    X_test = test_set.drop('class', axis=1)
    y_test = test_set['class']

    # X_train, y_train = random_train(X_train, y_train, n_sample)
    X_test = random_test_feature(X_test, n_sample)

    confu_matrix, result_report, accuracy = decision_tree(X_train, X_test, y_train, y_test, confu_csv, result_txt)

    # confu_matrix, result_report, accuracy = svm(X_train, X_test, y_train, y_test, confu_csv, result_txt)

    # Transform the str type in dataset to value so that can be trained with fit() function
    # X_train = label_encoder(X_train)
    # X_test = label_encoder(X_test)
    #
    # classifier = DecisionTreeClassifier()
    # classifier.fit(X_train, y_train)
    #
    # y_pred = classifier.predict(X_test)
    #
    # confu_matrix = confusion_matrix(y_test, y_pred)
    # pd.DataFrame(confu_matrix).to_csv(confu_csv)
    #
    # result_report = classification_report(y_test, y_pred)
    # with open(result_txt, 'w') as f:
    #     f.write(result_report)

    print("The confusion matrix is: {}\n Accuracy is:{}.".format(confu_matrix, accuracy))
    print("The classification report is located at: {}.\n {}".format(result_txt, result_report))







