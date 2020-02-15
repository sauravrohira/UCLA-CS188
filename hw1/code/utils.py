import os
import cv2
import numpy as np
import timeit, time
import random
from sklearn import neighbors, svm, cluster, preprocessing
from scipy.spatial import distance

def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path) if not dirname.startswith('.')], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path) if not dirname.startswith('.')], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    classifier = neighbors.KNeighborsClassifier(num_neighbors)
    classifier.fit(train_features, train_labels)
    predicted_categories = classifier.predict(test_features)
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.

    clf = svm.LinearSVC(random_state=0, C=svm_lambda)
    if is_linear == False:
        clf = svm.SVC(random_state=0, C=svm_lambda)
    clf.fit(train_features, train_labels)
    predicted_categories = clf.predict(test_features)
    return predicted_categories


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    dim = (target_size, target_size)
    resized_image = cv2.resize(input_image, dim)
    output_image = np.zeros(dim)
    output_image = cv2.normalize(resized_image, output_image, -1, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    return output_image


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)
    count_right = 0
    for i in range(0, len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            count_right += 1
    accuracy = (count_right/len(true_labels))*100
    return accuracy


def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.

    descriptors = []
    vocabulary  = []

    if feature_type == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        for img in train_images:
            _, descriptors_sift = sift.detectAndCompute(img, None)
            for desc in descriptors_sift:
                descriptors.append(desc)

    if feature_type == 'surf':
        surf = cv2.xfeatures2d.SURF_create()
        for img in train_images:
            _, descriptors_surf = surf.detectAndCompute(img, None)
            for desc in descriptors_surf:
                descriptors.append(desc)

    if feature_type == 'orb':
        orb = cv2.ORB_create()
        for img in train_images:
            _, descriptors_orb = orb.detectAndCompute(img, None)
            if not isinstance(descriptors_orb, np.ndarray):
                continue
            for o in descriptors_orb:
                descriptors.append(o)
    
    if clustering_type == 'kmeans':
        cluster_kmeans = cluster.KMeans(n_clusters=dict_size).fit(descriptors)
        vocabulary = cluster_kmeans.cluster_centers_
        
    if clustering_type == 'hierarchal':
        cluster_hierarchical = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(descriptors)

        desc_map = []
        for label in range(0, dict_size):
            desc_map.append([])
            for i in range(0, len(descriptors)):
                if cluster_hierarchical.labels_[i] == label:
                    desc_map[label].append(descriptors[i])

        for label in range(0, dict_size):
            average_desc = []
            for j in range(0, len(descriptors[0])):
                average_val = 0
                for descriptor in desc_map[label]:
                    average_val += descriptor[j]
                average_val /= len(desc_map[label])
                average_desc.append(average_val)
            vocabulary.append(average_desc)

    return vocabulary


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram

    descriptors = []
    dist_array = np.zeros(len(vocabulary))

    if feature_type == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        descriptors = np.array(sift.detectAndCompute(image, None)[1])

    if feature_type == 'surf':
        surf = cv2.xfeatures2d.SURF_create()
        surf_desc = surf.detectAndCompute(image, None)[1]
        for des in surf_desc:
            descriptors.append(des)
        descriptors = np.array(descriptors)

    if feature_type == 'orb':
        orb = cv2.ORB_create()
        orb_desc = orb.detectAndCompute(image, None)[1]
        for des in orb_desc:
            descriptors.append(des)
        descriptors = np.array(descriptors)

    knn = neighbors.KNeighborsClassifier(1)
    vocab_ind = list(range(0, len(vocabulary)))
    knn.fit(vocabulary, vocab_ind)

    closest_word = knn.predict(descriptors)

    for i in closest_word:
        dist_array[i] += 1

    Bow = dist_array
    for i, b in enumerate(Bow):
        Bow[i] = b/len(Bow)
    
    return Bow

def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds

    sizes = np.array([8, 16, 32])
    neighbors = np.array([1, 3, 6])
    classResult = []

    for size in sizes:
        resize_start_time = time.time()
        train_features_flat = []
        test_features_flat  = []
        for i in range(0, len(train_features)):
            train_features_flat.append(np.ndarray.flatten(imresize(train_features[i], size)))
        for j in range(0, len(test_features)):
            test_features_flat.append(np.ndarray.flatten(imresize(test_features[j], size)))
        resize_end_time = time.time()
        for k in neighbors:
            knn_start_time = time.time()
            predicted_labels = KNN_classifier(train_features_flat, train_labels, test_features_flat, k)
            knn_end_time = time.time()
            acc = reportAccuracy(test_labels, predicted_labels)
            runtime = ((resize_end_time - resize_start_time) + (knn_end_time - knn_start_time))
            print("size: " + str(size) + " k: " + str(k))
            print("accuracy: " + str(acc) + " runtime: " + str(runtime))
            classResult.append(acc)           
            classResult.append(runtime)

    return classResult
    
