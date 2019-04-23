import os
from Preprocessing.preProcess import *
from data.MNIST.full_data import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PCA.entropy import *
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
import tqdm


mnist_path = os.path.join("..", "data", "MNIST")

x_train_path = os.path.join(mnist_path, "full_data", "x_train.npy")
x_test_path = os.path.join(mnist_path, "full_data", "x_test.npy")
y_train_path = os.path.join(mnist_path, "full_data", "y_train.npy")
y_test_path = os.path.join(mnist_path, "full_data", "y_test.npy")

print("Loading data from %s" % x_train_path)
print("Loading data from %s" % x_test_path)
print("Loading location of %s" % y_train_path)
print("Loading location of %s" % y_test_path)

x_train = np.load(x_train_path)
x_test = np.load(x_test_path)

# pixels = x_train[0].reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()

# arrays to store data
explained_variances = [.99, .97, .95, .9, .8, .7, .6, .5, .4, .3, .2, .1, .05]
x_train_pcas = []
x_test_pcas = []
dimensions = []
entropies = []

# iterate through each desired variance
for variance in explained_variances:
    # initialize the pca with the desired variance
    pca = PCA(variance)
    # fit the pca around the training set
    pca.fit(x_train)
    # transform the set according to the fit (ACTUAL PCA TAKES PLACE HERE)
    x_train_transformed = pca.transform(x_train)
    x_train_transformed = x_train_transformed.astype(np.float32)
    approximation = pca.inverse_transform(x_train_transformed)

    x_test_transformed = pca.transform(x_test)
    x_test_transformed = x_test_transformed.astype(np.float32)
    # add the transformation and its values to the arrays
    x_train_pcas.append(x_train_transformed)
    dimensions.append(pca.n_components_)
    entropy = averageEntropy(x_train_transformed)
    entropies.append(entropy)
    analysisData = "dimensions,%s\n" \
                   "entropy,%s" %\
                   (pca.n_components_, entropy)
    # print("With %s of the explained variance, the data was reduced to %s principal components." % (variance, pca.n_components_))
    # print("The average entropy for the transformed set is %s." % entropy)

    # save the transformed set and analysis along the desired path
    save_path = os.path.join(mnist_path, "pca_data", "dims_" + str(pca.n_components_))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print("Saving transformed sets, pictures, and analysis in %s" % save_path)
    np.save(os.path.join(save_path, "x_train"), x_train_transformed)
    np.save(os.path.join(save_path, "x_test"), x_test_transformed)
    copyfile(y_train_path, os.path.join(save_path, "y_train.npy"))
    copyfile(y_test_path, os.path.join(save_path, "y_test.npy"))

    plt.imshow(approximation[0].reshape(28, 28),
               cmap=plt.cm.gray, interpolation='nearest',
               clim=(0, 255))
    plt.title("Reconstruction with %s principal components" % pca.n_components_)
    plt.savefig(os.path.join(save_path, "reconstruct.png"))

    file = open(os.path.join(save_path, "analysis.csv"), "w")
    file.write(analysisData)

# plt.title("Variance vs. Dimensions")
# plt.xlabel("Explained Variance")
# plt.ylabel("Principal Components (dimensions)")
# plt.plot(explained_variances, dimensions, 'bo-')
# plt.show()
#
# plt.title("Entropy vs. Dimensions")
# plt.xlabel("Entropy")
# plt.ylabel("Principal Components (dimensions)")
# plt.plot(entropies, dimensions, 'bo-')
# plt.show()
#
# plt.title("Variance vs. Entropy")
# plt.xlabel("Explained Variance (\%)")
# plt.ylabel("Entropy")
# plt.plot(explained_variances, entropies, 'bo-')
# plt.show()