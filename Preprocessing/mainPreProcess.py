from Preprocessing.preProcess import *
import os

save_path = "../data/"
read_path = "./winequality-red.csv"
if not os.path.exists(save_path):
        os.mkdir(save_path)

features, labels = readDataFromCSV(read_path)
train_features, test_features, validation_features = splitData(features)
train_labels, test_labels, validation_labels = splitData(labels)
writeToCSV(train_features,save_path + "train_features.csv")
writeToCSV(test_features,save_path + "test_features.csv")
writeToCSV(validation_features,save_path + "valid_features.csv")
writeToCSV(train_labels,save_path + "train_labels.csv")
writeToCSV(test_labels,save_path + "test_labels.csv")
writeToCSV(validation_labels,save_path + "valid_lables.csv")
