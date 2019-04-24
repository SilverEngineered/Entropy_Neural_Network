import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PCA.entropy import *
from PCA.utils import *
from shutil import copyfile
from tqdm import tqdm

mnist_path = os.path.join("..", "data", "MNIST")

x_train_path = os.path.join(mnist_path, "full_data", "x_train.npy")
x_test_path = os.path.join(mnist_path, "full_data", "x_test.npy")
y_train_path = os.path.join(mnist_path, "full_data", "y_train.npy")
y_test_path = os.path.join(mnist_path, "full_data", "y_test.npy")

x_train = np.load(x_train_path)
x_test = np.load(x_test_path)

# pixels = x_train[0].reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()
#
# pixels = x_train[1].reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()
#
# pixels = x_train[2].reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()
#
# pixels = x_train[3].reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()

# arrays to store data
explained_variances = [.99, .97, .95, .9, .8, .7, .5, .05]
# explained_variances = [.99]
x_train_pcas = []
x_test_pcas = []
dimensions = []
entropies = []

# # iterate through each desired variance
# for variance in tqdm(explained_variances):
#     # initialize the pca with the desired variance
#     pca = PCA(variance)
#     # fit the pca around the training set
#     pca.fit(x_train)
#     # transform the set according to the fit (ACTUAL PCA TAKES PLACE HERE)
#     x_train_transformed = pca.transform(x_train)
#     x_train_transformed = x_train_transformed.astype(np.float32)
#     approximation = pca.inverse_transform(x_train_transformed)
#
#     x_test_transformed = pca.transform(x_test)
#     x_test_transformed = x_test_transformed.astype(np.float32)
#     # add the transformation and its values to the arrays
#     x_train_pcas.append(x_train_transformed)
#     dimensions.append(pca.n_components_)
#     entropy = averageEntropy(x_train_transformed)
#     entropies.append(entropy)
#     analysisData = "dimensions,%s\n" \
#                    "entropy,%s\n" \
#                    "variance,%s\n" %\
#                    (pca.n_components_, entropy, variance)
#     # print("With %s of the explained variance, the data was reduced to %s principal components." % (variance, pca.n_components_))
#     # print("The average entropy for the transformed set is %s." % entropy)
#
#     # save the transformed set and analysis along the desired path
#     save_path = os.path.join(mnist_path, "pca_data", "dims_" + str(pca.n_components_))
#     if not os.path.exists(save_path):
#         os.mkdir(save_path)
#     # print("Saving transformed sets, pictures, and analysis in %s" % save_path)
#     np.save(os.path.join(save_path, "x_train"), x_train_transformed)
#     np.save(os.path.join(save_path, "x_test"), x_test_transformed)
#     copyfile(y_train_path, os.path.join(save_path, "y_train.npy"))
#     copyfile(y_test_path, os.path.join(save_path, "y_test.npy"))
#
#     plt.imshow(approximation[0].reshape(28, 28),
#                cmap=plt.cm.gray, interpolation='nearest',
#                clim=(0, 255))
#     plt.title("Reconstruction with %s principal components" % pca.n_components_)
#     plt.savefig(os.path.join(save_path, "reconstruct.png"))
#     plt.close()
#
#     file = open(os.path.join(save_path, "analysis.csv"), "w")
#     file.write(analysisData)

generateGraphs()