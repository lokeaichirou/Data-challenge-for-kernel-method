from __future__ import division, print_function
import numpy as np
import cvxopt
import time
from scipy.sparse import csr_matrix, dok_matrix

# Hide cvxopt output
cvxopt.solvers.options['show_progress'] = False


# Loading data

def loadData(fileName_X, fileName_Y=None):
    '''
    Loading files
    :param fileName: path
    :return: dataset and labels
    '''

    # Collect files
    dataArr = np.loadtxt(fileName_X, skiprows=1, usecols=(1,), dtype=str, delimiter=',')
    labelArr = []
    if fileName_Y is not None:
        labelArr = np.loadtxt(fileName_Y, skiprows=1, usecols=(1,), dtype=str, delimiter=',')
        labelArr = [2 * int(x) - 1 for x in labelArr]

    # return dataset and labels
    return dataArr, labelArr


# string embedding

def base2int(c):
    return {'A': 0, 'C': 1, 'G': 2, 'T': 3}.get(c, 0)


def string_encoder(sample_strings):
    encoded_strings = []

    for sample_index, sample_string in enumerate(sample_strings):
        string = [base2int(c) for c in sample_string]
        encoded_strings.append(string)

    encoded_strings = np.array(encoded_strings)
    return encoded_strings


def kmer_to_spectral_index_multiplier(kmer_size, alphabet_size=4):
    return alphabet_size ** np.arange(kmer_size)


def kmer_decomposition(sample_strings, kmer_size=8):

    n_rows = len(sample_strings)
    n_features = 4 ** kmer_size
    n_features = int(n_features)
    spectral_embedding = dok_matrix((n_rows, n_features))
    base4_multiplier = kmer_to_spectral_index_multiplier(kmer_size)

    for sample_index, sample_string in enumerate(sample_strings):
        input_bases = np.array([base2int(c) for c in sample_string])
        string_length = len(sample_string)
        for i in range(string_length - kmer_size + 1):
            kmer = input_bases[i:i + kmer_size]
            kmer_index = base4_multiplier.dot(kmer)
            spectral_embedding[sample_index, kmer_index] += 1
    return csr_matrix(spectral_embedding).toarray()


# Gaussian kernel with parameters
def rbf_kernel(x1, x2, sigma):
    X12norm = x1.dot(x1) - 2 * x1.dot(x2) + x2.dot(x2)
    return np.exp(-X12norm / (2 * sigma ** 2))


class SupportVectorMachine(object):
    """The Support Vector Machine classifier.
    Uses cvxopt to solve the quadratic optimization problem.
    Parameters:
    -----------
    C: float
        Penalty term.
    kernel: function
        Kernel function. Can be either polynomial, rbf or linear.
    sigma: float
        Used in the Gaussian kernel function.
    """

    def __init__(self, trainDataList, trainLabelList, kernel=rbf_kernel, C=1, power=2, sigma=None):
        self.trainDataList = trainDataList
        self.trainLabelList = trainLabelList
        self.kernel = kernel
        self.C = C
        self.power = power
        self.sigma = sigma
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self):

        n_samples, n_features = np.shape(self.trainDataList)

        # Set gamma to 1/n_features by default
        if not self.sigma:
            self.sigma = 1 / n_features

        # Calculate kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(self.trainDataList[i], self.trainDataList[j], self.sigma)

        # Define the quadratic optimization problem
        P = cvxopt.matrix(np.outer(self.trainLabelList, self.trainLabelList) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(self.trainLabelList, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagr_mult > 1e-8
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = self.trainDataList[idx]
        # Get the corresponding labels
        self.support_vector_labels = np.array(self.trainLabelList)[idx]

        # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0], self.sigma)

    def predict(self, X):
        # Iterate through list of samples and make predictions
        prediction = 0
        w = 0
        # Determine the label of the sample by the support vectors
        for i in range(len(self.lagr_multipliers)):
            w += self.lagr_multipliers[i] * self.support_vector_labels[i] \
                 * self.support_vectors[i]
        prediction = np.sign(np.array(w).dot(X) + self.intercept)
        return prediction

    def validate(self):
        '''
        Validation
        :param testDataList: Validation Dataset
        :param testLabelList: Validation Labels
        :return: correnctency
        '''
        # error count
        errorCnt = 0
        # iterate all validation samples
        for i in range(len(self.trainDataList)):
            # print progress
            print('validate:%d:%d' % (i, len(self.trainDataList)))
            result = self.predict(self.trainDataList[i])
            if result != self.trainLabelList[i]:
                errorCnt += 1
        return 1 - errorCnt / len(self.trainDataList)

    def test_and_generate(self, testDataList):
        y_test = []
        # iterate all testing samples
        for i in range(len(testDataList)):
            print('test:%d:%d' % (i, len(testDataList)))
            result = self.predict(testDataList[i])
            y_test.append(result)

        y_test = np.array(y_test)
        y_test[y_test == -1] = 0
        y_save = np.vstack([np.arange(len(y_test)), y_test]).T
        np.savetxt('kernel_svm.csv', y_save, delimiter=',', header='Id,Bound', fmt='%i', comments='')


if __name__ == '__main__':
    start = time.time()

    X0, Y0 = loadData('data/Xtr0.csv', 'data/Ytr0.csv')
    X1, Y1 = loadData('data/Xtr1.csv', 'data/Ytr1.csv')
    X2, Y2 = loadData('data/Xtr2.csv', 'data/Ytr2.csv')
    X_train = np.hstack((X0, X1, X2))
    print(len(X_train))
    Y_train = np.hstack((Y0, Y1, Y2))
    print(len(Y_train))

    X_test_0, _ = loadData('data/Xte0.csv')
    X_test_1, _ = loadData('data/Xte1.csv')
    X_test_2, _ = loadData('data/Xte2.csv')
    X_test = np.hstack((X_test_0, X_test_1, X_test_2))
    #X_train = string_encoder(X_train)
    #X_test = string_encoder(X_test)

    X_train = kmer_decomposition(X_train, kmer_size=8)
    X_test = kmer_decomposition(X_test, kmer_size=8)

    # Initialize SVM
    print('start init SVM')
    svm = SupportVectorMachine(X_train, Y_train, kernel=rbf_kernel, C=1, power=2, sigma=None)

    # Training
    print('start to train')
    svm.fit()

    # Validation
    print('start to validate')
    accuracy = svm.validate()
    print('the accuracy is:%d' % (accuracy * 100), '%')

    # Prediction and Testing
    svm.test_and_generate(X_test)

    # Time spent
    print('time span:', time.time() - start)
