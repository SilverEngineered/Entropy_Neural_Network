import os
import numpy as np
import matplotlib.pyplot as plt
import csv

mnist_path = os.path.join("..", "data", "MNIST")

def generateGraphs(dimensions=None, entropies=None, variances=None):
    if (dimensions != None) and (variances != None):
        plt.title("Variance vs. Dimensions")
        plt.xlabel("Explained Variance")
        plt.ylabel("Principal Components (dimensions)")
        plt.plot(variances, dimensions, 'bo-')
        plt.show()

    if (entropies != None) and (dimensions != None):
        plt.title("Entropy vs. Dimensions")
        plt.xlabel("Entropy")
        plt.ylabel("Principal Components (dimensions)")
        plt.plot(entropies, dimensions, 'bo-')
        plt.show()

    if (entropies != None) and (variances != None):
        plt.title("Variance vs. Entropy")
        plt.xlabel("Explained Variance (\%)")
        plt.ylabel("Entropy")
        plt.plot(variances, entropies, 'bo-')
        plt.show()

    plt.title("Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    # baseAccCSV = open(os.path.join(mnist_path, "full_data", "nnacc1.csv"))
    # plt.plot(np.arange(200), list(map(float, baseAccCSV.read().split('\n')[:200])),
    #          '-', label='784')
    for root, dirs, files in os.walk(os.path.join(mnist_path, "pca_data"), topdown=True):
        if "dims_" in root:
            print(root)
            label = ""
            reader = csv.reader(open(os.path.join(root, "analysis.csv")))
            for row in reader:
                if row[0] == 'dimensions':
                    label = row[1]
            iterationsAcc = []
            for file in files:
                if "nnacc" in file:
                    accCSV = open(os.path.join(root, file))
                    iterationsAcc.append(list(map(float, accCSV.read().split('\n')[:200])))

            finalAvgAcc = []
            for _ in range(200):
                finalAvgAcc.append(0)
            for iteration in iterationsAcc:
                for count, val in enumerate(iteration):
                    finalAvgAcc[count] = finalAvgAcc[count] + val
            for count, val in enumerate(finalAvgAcc):
                finalAvgAcc[count] = finalAvgAcc[count] / len(iterationsAcc)
            plt.plot(np.arange(0,200), finalAvgAcc, "-", label=label)
    plt.legend(title="Dimensions")
    plt.show()